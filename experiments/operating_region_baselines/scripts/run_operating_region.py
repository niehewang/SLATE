from __future__ import annotations
import argparse, collections, csv, gc, json, math, shutil, sys, traceback
from pathlib import Path
import numpy as np
from common import *
SHARED_HELPERS=Path(__file__).resolve().parents[2]/'_shared'
if str(SHARED_HELPERS) not in sys.path: sys.path.insert(0,str(SHARED_HELPERS))
import core_helpers as C

BASE_METHODS=('M0_Unwrapped','M1_MarginalCover','M2_ScheduleOnly')
FULL_LAMBDAS=(0.05,0.1,0.2,0.5,1.0)
ALL_CONDITIONS=BASE_METHODS+tuple(f'M3_Full_{l:g}' for l in FULL_LAMBDAS)

def log(rd,msg):
    s=str(msg); print(s,flush=True)
    with open(Path(rd)/'run.log','a',encoding='utf-8') as f:f.write(s+'\n')

def dump(ret,name,obj): atomic_json(Path(ret)/name,obj)

def configure_core(rd):
    rd=Path(rd); C.CFG=PCFG; C.RUN=rd/'work'; C.RET=rd/'work'/'legacy_return'; C.BANK=rd/'work'/'banks'; C.EMB=rd/'work'/'embeddings'; C.CACHE=rd/'work'/'cache'; C.LOG=rd/'run.log'
    for p in [C.RUN,C.RET,C.BANK,C.EMB,C.CACHE]:p.mkdir(parents=True,exist_ok=True)
    C.P3A=Path(CFG['llmdna_root']).parents[1]; C.LLMDNA_SRC=Path(CFG['llmdna_root'])/'src'
    C.QWEN=Path(CFG['qwen_source']); C.PHI=Path(CFG['phi_source']); C.ENC=Path(CFG['qwen_embedding']); C.MPNET=Path(CFG['mpnet']); C.TRAFFIC=Path(CFG['traffic'])
    if str(C.LLMDNA_SRC) not in sys.path:sys.path.insert(0,str(C.LLMDNA_SRC))

def load_pool_jsonl(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip():out.append(json.loads(line))
    return out

def prior_exclusion(): return json.load(open(Path(CFG['prior_heldout_exclusion']),encoding='utf-8'))

def reconstructed_devcal():
    traffic=Path(CFG['traffic']); ds=sessions_for_split(traffic,'development','P3B_development'); cs=sessions_for_split(traffic,'calibration','P3B_calibration')
    return ds,cs,freeze_pool(ds,512,'P3B_DEV_POOL'),freeze_pool(cs,128,'P3B_CAL_POOL')

def fresh_held_real(g4_pool):
    traffic=Path(CFG['traffic']); ds=sessions_for_split(traffic,'development','P3B_development'); cs=sessions_for_split(traffic,'calibration','P3B_calibration'); hs=sessions_for_split(traffic,'heldout','P3C_heldout')
    dev=freeze_pool(ds,512,'P3B_DEV_POOL'); cal=freeze_pool(cs,128,'P3B_CAL_POOL'); old=freeze_pool(hs,128,'P3C_HELDOUT_POOL'); prior=prior_exclusion()
    g3h=set(prior['conversation_id_sha256']); banned={x['conversation_id'] for x in dev+cal+old}|{x['conversation_id'] for x in g4_pool}
    return [s for s in hs if s['id'] not in banned and sha_text(str(s['id'])) not in g3h]

def prior_bank_rows(tag,records):
    p=Path(CFG['p3b_run'])/'banks'/f'{tag}.json'
    if not p.exists():return None
    try:state=json.load(open(p,encoding='utf-8')); mp={r['id']:r for r in state.get('rows',[]) if r.get('response') is not None}
    except Exception:return None
    if not all(r['id'] in mp and mp[r['id']].get('text')==r['text'] for r in records):return None
    return [mp[r['id']] for r in records]

def acquire_devcal_banks(records):
    audit={}; banks={}
    for tag,path,kind in [('qwen_source',C.QWEN,'source'),('qwen_nf4',C.QWEN,'nf4'),('phi_source',C.PHI,'phi')]:
        old=prior_bank_rows(tag,records)
        if old is not None:
            banks[tag]=old; audit[tag]={'source':'reused_exact_prior_response_asset','prior_path':str(Path(CFG['p3b_run'])/'banks'/f'{tag}.json'),'n':len(old),'query_id_sha256':sha_text('\n'.join(r['id'] for r in old))}; atomic_json(C.BANK/f'{tag}.json',{'tag':tag,'kind':kind,'rows':old})
        else:
            banks[tag]=C.generate_bank(tag,path,kind,records); audit[tag]={'source':'regenerated_exact_frozen_model_and_generation_settings','n':len(banks[tag]),'query_id_sha256':sha_text('\n'.join(r['id'] for r in banks[tag]))}
    return banks,audit

def embed_banks(banks):
    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer(str(C.ENC),device='cuda:0'); E={k:C.embed_bank(k,v,enc) for k,v in banks.items()}; del enc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:pass
    return E

def source_only_J(qwen,phi,scale=None):
    raw=1-C.cosrow(np.asarray(qwen)[:,:64],np.asarray(phi)[:,:64])
    if scale is None: lo,hi=np.quantile(raw,[.05,.95]); scale=(float(lo),float(hi))
    lo,hi=scale; return np.clip((raw-lo)/max(float(hi-lo),1e-8),0,1),scale,raw

def paired_seed(split_tag,episode): return int(PCFG['seed']+C.h64(f'{split_tag}|{episode}','G4_PAIRED_SESSION')%100000000)

def sample_by_weights(weights,q,seed):
    w=np.asarray(weights,float); w=np.maximum(w,1e-300); w=w/w.sum(); rng=np.random.default_rng(seed); g=-np.log(-np.log(np.clip(rng.random(len(w)),1e-12,1-1e-12))); idx=np.argsort(np.log(w)+g)[-q:]
    return idx.astype(int).tolist(),w

def state_marginal_from_dev(sched,dev_real):
    S=int(sched['S']); cnt=np.ones(S,dtype=float)*1e-3
    for s in dev_real:
        st=sched['state_fn'](s['texts'])
        for x in st:cnt[int(x)]+=1
    p=cnt/cnt.sum(); return p,{'S':S,'smoothing_pseudocount_per_state':1e-3,'counts_plus_smoothing':cnt.tolist(),'probabilities':p.tolist(),'source':'development benign traffic only'}

def condition_sessions(kind,pool,J,states,sched,state_marginal,n,q,split_tag,lam=None):
    if kind in ('M0_Unwrapped','M2_ScheduleOnly'):weights=np.ones(len(pool),float)/len(pool)
    elif kind=='M1_MarginalCover':weights=np.asarray(state_marginal)[np.asarray(states,int)]; weights=weights/weights.sum()
    elif kind=='M3_Full':weights=C.distribution_weights(J,float(lam))
    else:raise ValueError(kind)
    out=[]
    for e in range(n):
        seed=paired_seed(split_tag,e); sel,w=sample_by_weights(weights,q,seed); rng=np.random.default_rng(seed+17); rnd=np.asarray(sel)[rng.permutation(q)].astype(int).tolist()
        scheduled=kind in ('M2_ScheduleOnly','M3_Full'); order=sched['schedule'](sel,states,seed+31) if scheduled else rnd
        out.append({'episode':e,'seed':seed,'selected':sel,'random_order':rnd,'order':list(map(int,order)),'mean_J':float(np.mean(np.asarray(J)[sel])) if kind=='M3_Full' else None})
    return out,weights

def train_bundle(tag,dev_real,dev_pool,dev_session_groups,mp):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ec=C.ECache(mp); posq=[]
    for ss in dev_session_groups:posq.extend(C.flatten_queries(dev_pool,ss))
    posq=posq[:PCFG['detector_query_cap']]; negq=C.excluded_queries([{'id':s['id'],'texts':s['texts']} for s in dev_real],[r['text'] for r in dev_pool],PCFG['detector_query_cap'],'G4_'+tag+'_DNEG')
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=100000,sublinear_tf=True,strip_accents='unicode'); X=vec.fit_transform(posq+negq); y=np.r_[np.ones(len(posq)),np.zeros(len(negq))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=PCFG['seed']).fit(X,y); d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=PCFG['seed']).fit(np.r_[ec.get(posq),ec.get(negq)],y)
    pooled=[s for ss in dev_session_groups for s in ss]; seq={}; real=[{'id':s['id'],'texts':s['texts']} for s in dev_real]
    offs={'M0':0,'M1':1000,'M2':2000,'M3':3000}.get(tag,4000)
    for L in PCFG['L_grid']:
        pos=C.audit_items(dev_pool,pooled,L)[:PCFG['detector_window_cap']]; negw=C.real_windows(real,L,PCFG['detector_window_cap'],'G4_'+tag+'_NEGW'+str(L))
        if L in (4,8):seq[f'D3_L{L}']=C.fit_seq(ec,pos,negw,L,PCFG['d3_hidden'],False,PCFG['seed']+offs+L)
        seq[f'D4_L{L}']=C.fit_seq(ec,pos,negw,L,PCFG['d4_hidden'],True,PCFG['seed']+offs+100+L)
    return {'d1':d1,'d2':d2,'vec':vec,'ec':ec,'seq':seq}

def calibration_neg(bundle,cal_real,cal_pool,tag):
    real=[{'id':s['id'],'texts':s['texts']} for s in cal_real]; negq=C.excluded_queries(real,[r['text'] for r in cal_pool],6000,'G4_'+tag+'_CALNEG')
    if len(negq)<int(CFG['min_negative_queries']):raise RuntimeError(f'insufficient calibration negatives for {tag}: {len(negq)}')
    out={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(negq))[:,1],'D2':bundle['d2'].predict_proba(bundle['ec'].get(negq))[:,1]}
    for L in PCFG['L_grid']:
        nw=C.real_windows(real,L,PCFG['detector_window_cap'],'G4_'+tag+'_CALW'+str(L))
        if L in (4,8):out[f'D3_L{L}']=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],nw,L)
        out[f'D4_L{L}']=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],nw,L)
    return out

def positive_scores(bundle,pool,sessions):
    qs=C.flatten_queries(pool,sessions); out={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(qs))[:,1],'D2':bundle['d2'].predict_proba(bundle['ec'].get(qs))[:,1]}
    for L in PCFG['L_grid']:
        ai=C.audit_items(pool,sessions,L)
        if L in (4,8):out[f'D3_L{L}']=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],ai,L)
        out[f'D4_L{L}']=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],ai,L)
    return out

def candidates_for_L(L):return ['D1','D2','D4_L1'] if int(L)==1 else [f'D3_L{L}',f'D4_L{L}']

def metric_summary(bundle,pool,sessions,neg):
    pos=positive_scores(bundle,pool,sessions); strongest={}; allm={}
    for L in PCFG['L_grid']:
        cand=candidates_for_L(L); mm={k:C.roc_metrics(pos[k],neg[k]) for k in cand}; best=max(cand,key=lambda k:mm[k]['AUROC_sym']); strongest[str(L)]={'detector':best,**mm[best]}; allm[str(L)]=mm
    return strongest,allm,pos

def frontier(pos,neg):
    from sklearn.metrics import roc_curve
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; s=np.r_[np.asarray(pos),np.asarray(neg)]; fpr,tpr,thr=roc_curve(y,s)
    return [{'AIR':float(a),'BDR':float(b),'threshold':float(t) if np.isfinite(t) else str(t)} for a,b,t in zip(tpr,fpr,thr)]

def calibrate_lineage(cal_sessions,calE):
    ad=C.DNAAdapter(); rows=[C.lineage_session(ad,s['order'],calE['qwen_nf4'],calE['qwen_source'],calE['phi_source']) for s in cal_sessions]
    neg=np.asarray([r['sim_phi'] for r in rows]); tau=float(np.quantile(neg,.95,method='higher')); rt=float(np.quantile([r['risk'] for r in rows],.975,method='higher')); gt=float(np.quantile([r['gap'] for r in rows],.025,method='lower'))
    return {'similarity_threshold':tau,'risk_threshold_kappa':rt,'bank_gap_threshold':gt,'calibration_n':len(rows),'false_source_calibration_rate':float(np.mean(neg>=tau)),'procedure':'same frozen uniform-calibration lineage/risk procedure; numeric values frozen before held-out'}

def verdict_composition(rows):
    n=max(1,len(rows)); d=collections.Counter()
    for r in rows:
        if r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base':d['correct-source verified']+=1
        elif r['verdict']=='Verified':d['false-source verified']+=1
        elif r['verdict']=='Inconclusive-Serving':d['Channel-Risk']+=1
        elif r['verdict']=='Inconclusive-Bank':d['Insufficient-Evidence']+=1
        elif r['verdict']=='Rejected':d['Rejected']+=1
        else:d['Other']+=1
    return {k:{'count':int(d[k]),'rate':float(d[k]/n)} for k in ['correct-source verified','false-source verified','Channel-Risk','Insufficient-Evidence','Rejected','Other']}

def detector_freeze(bundle,cal_pool,cal_sessions,cal_neg):
    strongest,allm,pos=metric_summary(bundle,cal_pool,cal_sessions,cal_neg); thresholds={}; keys=['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']
    for beta in PCFG['beta_grid']:
        thresholds[str(beta)]={}
        for k in keys:
            th,b=C.threshold_for_bdr(cal_neg[k],beta); thresholds[str(beta)][k]={'threshold':float(th),'measured_BDR':float(b)}
    stat=max(['D1','D2'],key=lambda k:C.roc_metrics(pos[k],cal_neg[k])['AUROC_sym']); state=max(['D3_L4','D4_L4'],key=lambda k:C.roc_metrics(pos[k],cal_neg[k])['AUROC_sym'])
    return {'strongest_by_L':strongest,'all_calibration_metrics':allm,'thresholds':thresholds,'stateless_attacker':stat,'stateful_attacker_L4':state}

def full_theory(J,sessions,lam):
    w=C.distribution_weights(J,lam); K=float(PCFG['main_q']*np.sum(w*np.log(np.maximum(w*len(w),1e-300)))); meanj=float(np.mean([s['mean_J'] for s in sessions]))
    return {'mean_J':meanj,'pool_mean_J':float(np.mean(J)),'Gamma_select':meanj-float(np.mean(J)),'Khat_q':K}

def overlap_diag(pool,sessions,J):
    sets=[set(s['selected']) for s in sessions]; conv=[set(pool[i]['conversation_id'] for i in s['selected']) for s in sessions]; inter=[]; jac=[]; cinter=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            x=len(sets[i]&sets[j]); inter.append(x); jac.append(x/max(1,len(sets[i]|sets[j]))); cinter.append(len(conv[i]&conv[j]))
    freq=np.zeros(len(pool),int)
    for s in sessions:freq[np.asarray(s['selected'],int)]+=1
    qs=np.quantile(J,np.linspace(0,1,11)); dec=[]
    for d in range(10):
        lo,hi=qs[d],qs[d+1]; mask=(J>=lo)&((J<=hi) if d==9 else (J<hi)); dec.append({'decile':d+1,'queries':int(mask.sum()),'inclusions':int(freq[mask].sum()),'mean_inclusion':float(freq[mask].mean()) if mask.any() else 0.0})
    stat=lambda a:{'mean':float(np.mean(a)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(np.max(a))} if a else {}
    return {'unique_queries_used':int(np.sum(freq>0)),'pairwise_intersection':stat(inter),'pairwise_jaccard':stat(jac),'pairwise_conversation_intersection':stat(cinter),'query_score_decile_concentration':dec,'per_query_inclusion_frequency':{pool[i]['id']:int(x) for i,x in enumerate(freq) if x}}

def selection_weights(kind,J,states,state_marginal,lam=None):
    if kind in ('M0_Unwrapped','M2_ScheduleOnly'):return np.ones(len(J),float)/len(J)
    if kind=='M1_MarginalCover':
        w=np.asarray(state_marginal)[np.asarray(states,int)]; return w/w.sum()
    if kind=='M3_Full':return C.distribution_weights(J,float(lam))
    raise ValueError(kind)

def session_manifest_rows(cond,pool,sessions,J,states,state_marginal,kind,lam=None):
    w=selection_weights(kind,J,states,state_marginal,lam); rows=[]
    for s in sessions:
        sel=s['selected']; pre=s['random_order']; order=s['order']
        rows.append({'session_id':f'{cond}_{s["episode"]:03d}','method_condition':cond,'method_family':kind,'lambda':lam,'rng_seed':int(s['seed']),'query_ids':[pool[i]['id'] for i in sel],'conversation_ids':[pool[i]['conversation_id'] for i in sel],
          'source_only_J_diagnostic':[float(J[i]) for i in sel] if kind=='M3_Full' else None,'selection_weights':[float(w[i]) for i in sel],
          'pre_scheduler_order_query_ids':[pool[i]['id'] for i in pre],'transmitted_order_query_ids':[pool[i]['id'] for i in order],
          'lineage_evidence_order':'pre_scheduler_order' if kind in ('M2_ScheduleOnly','M3_Full') else 'transmitted_order',
          'J_E_used_for_selection':kind=='M3_Full','marginal_state_only':kind=='M1_MarginalCover'})
    return rows

def heldout_neg_scores(bundle,held_real,pool,tag):
    real=[{'id':s['id'],'texts':s['texts']} for s in held_real]; negq=C.excluded_queries(real,[r['text'] for r in pool],6000,'G4_'+tag+'_HELD_NEG')
    if len(negq)<int(CFG['min_negative_queries']):raise RuntimeError(f'insufficient fresh heldout benign negatives for {tag}: {len(negq)}')
    out={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(negq))[:,1],'D2':bundle['d2'].predict_proba(bundle['ec'].get(negq))[:,1]}
    for L in PCFG['L_grid']:
        nw=C.real_windows(real,L,PCFG['detector_window_cap'],'G4_'+tag+'_HELD_W'+str(L))
        if L in (4,8):out[f'D3_L{L}']=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],nw,L)
        out[f'D4_L{L}']=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],nw,L)
    return out

def heldout_neg_blocks(bundle,held_real,pool,tag):
    ex={r['text'] for r in pool}; seen=set(); qrows=[]
    for ss in held_real:
        for t in ss['texts']:
            if t in ex or t in seen:continue
            seen.add(t); qrows.append((t,ss['id']))
    qrows.sort(key=lambda x:C.h64(x[0]+'|'+x[1],'G4_'+tag+'_HELD_NEG')); qrows=qrows[:6000]
    if len(qrows)<int(CFG['min_negative_queries']):raise RuntimeError(f'insufficient fresh benign negative blocks for {tag}: {len(qrows)}')
    keys=['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']; blocks={k:collections.defaultdict(list) for k in keys}; texts=[x[0] for x in qrows]; cids=[x[1] for x in qrows]
    qscore={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(texts))[:,1],'D2':bundle['d2'].predict_proba(bundle['ec'].get(texts))[:,1]}; items=[([t],cid) for t,cid in qrows]; qscore['D4_L1']=C.score_seq(bundle['seq']['D4_L1'],bundle['ec'],items,1)
    for k,a in qscore.items():
        for cid,v in zip(cids,a):blocks[k][cid].append(float(v))
    real=[{'id':s['id'],'texts':s['texts']} for s in held_real]
    for L in (4,8):
        items=C.real_windows(real,L,PCFG['detector_window_cap'],'G4_'+tag+'_HELD_W'+str(L))
        a=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],items,L)
        for (_,cid),v in zip(items,a):blocks[f'D3_L{L}'][cid].append(float(v))
        a=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],items,L)
        for (_,cid),v in zip(items,a):blocks[f'D4_L{L}'][cid].append(float(v))
    allc=sorted({cid for d in blocks.values() for cid in d}); return {k:{cid:np.asarray(v,float) for cid,v in d.items()} for k,d in blocks.items()},allc

def sample_neg_blocks(blocks,sampled):
    out={}
    for k,d in blocks.items():
        z=[d[c] for c in sampled if c in d and len(d[c])]; out[k]=np.concatenate(z) if z else np.empty(0,float)
    return out

def metric_by_freeze(bundle,pool,sessions,neg,freeze):
    pos=positive_scores(bundle,pool,sessions); out={}; fronts={}
    for L in PCFG['L_grid']:
        det=freeze['strongest_by_L'][str(L)]['detector']; out[str(L)]={'detector':det,**C.roc_metrics(pos[det],neg[det])}; fronts[str(L)]=frontier(pos[det],neg[det])
    return out,fronts

def riskaware(kind):return kind!='M0_Unwrapped'
def evidence_order(kind,s):return s['random_order'] if kind in ('M2_ScheduleOnly','M3_Full') else s['order']

def a1_target(session,pool,bundle,strong,thr,nf4,phi):
    idx=session['order']; texts=[pool[i]['text'] for i in idx]; sc=bundle['d1'].predict_proba(bundle['vec'].transform(texts))[:,1] if strong=='D1' else bundle['d2'].predict_proba(bundle['ec'].get(texts))[:,1]
    route=np.asarray(sc)>=thr; mix=nf4.copy(); sel=np.asarray(idx,int); mix[sel[route]]=phi[sel[route]]; return mix,route,np.asarray(sc,float)

def a2_target(session,pool,bundle,strong,thr,nf4,phi,L=4):
    idx=session['order']; texts=[pool[i]['text'] for i in idx]; route=np.zeros(len(idx),bool); scores=[]
    for t in range(L-1,len(idx)):
        sc=float(C.score_seq(bundle['seq'][strong],bundle['ec'],[(texts[t-L+1:t+1],'x')],L)[0]); scores.append(sc); route[t]=sc>=thr
    mix=nf4.copy(); sel=np.asarray(idx,int); mix[sel[route]]=phi[sel[route]]; return mix,route,np.asarray(scores,float)

def random_exact_k(session,route_mask,nf4,phi,seed,eligible_from=0):
    idx=np.asarray(session['order'],int); eligible=np.arange(eligible_from,len(idx),dtype=int); k=int(np.sum(route_mask))
    if k>len(eligible):raise RuntimeError('exact-k random control impossible')
    rr=np.random.default_rng(seed); choose=rr.choice(eligible,k,replace=False) if k else np.asarray([],int); rmask=np.zeros(len(idx),bool); rmask[choose]=True; mix=nf4.copy(); mix[idx[rmask]]=phi[idx[rmask]]
    return mix,rmask

def random_seed(cond,beta,episode,attack):return int(PCFG['seed']+C.h64(f'{cond}|{beta}|{episode}|{attack}','G4_RANDOM_ROUTING')%100000000)

def summarize_rows(rows):return {**C.verdict_rates(rows),'composition':verdict_composition(rows)}

def eval_utility_sweep(cond,kind,sessions,pool,E,bundle,freeze,lin,heldneg):
    ad=C.DNAAdapter(); rt=lin['risk_threshold_kappa'] if riskaware(kind) else None; gt=lin['bank_gap_threshold'] if riskaware(kind) else None; tau=lin['similarity_threshold']
    honest=[]
    for s in sessions:honest.append(C.lineage_session(ad,evidence_order(kind,s),E['qwen_nf4'],E['qwen_source'],E['phi_source'],rt,gt,tau))
    out={'honest':summarize_rows(honest),'beta':{},'_rows':{'honest':honest}}
    for beta in PCFG['beta_grid']:
        bk=str(beta); stat=freeze['stateless_attacker']; state=freeze['stateful_attacker_L4']; th1=freeze['thresholds'][bk][stat]['threshold']; th2=freeze['thresholds'][bk][state]['threshold']
        rows={'A1_targeted':[],'A1_random':[],'A2_targeted':[],'A2_random':[]}; air={'A1_targeted':[],'A1_random':[],'A2_targeted':[],'A2_random':[]}; kdiff={'A1':[],'A2':[]}
        for s in sessions:
            eo=evidence_order(kind,s)
            mix,mask,_=a1_target(s,pool,bundle,stat,th1,E['qwen_nf4'],E['phi_source']); rseed=random_seed(cond,beta,s['episode'],'A1'); rmix,rmask=random_exact_k(s,mask,E['qwen_nf4'],E['phi_source'],rseed,0)
            rows['A1_targeted'].append(C.lineage_session(ad,eo,mix,E['qwen_source'],E['phi_source'],rt,gt,tau)); rows['A1_random'].append(C.lineage_session(ad,eo,rmix,E['qwen_source'],E['phi_source'],rt,gt,tau)); air['A1_targeted'].append(float(mask.mean())); air['A1_random'].append(float(rmask.mean())); kdiff['A1'].append(int(mask.sum()-rmask.sum()))
            mix,mask,_=a2_target(s,pool,bundle,state,th2,E['qwen_nf4'],E['phi_source'],4); rseed=random_seed(cond,beta,s['episode'],'A2'); rmix,rmask=random_exact_k(s,mask,E['qwen_nf4'],E['phi_source'],rseed,3)
            rows['A2_targeted'].append(C.lineage_session(ad,eo,mix,E['qwen_source'],E['phi_source'],rt,gt,tau)); rows['A2_random'].append(C.lineage_session(ad,eo,rmix,E['qwen_source'],E['phi_source'],rt,gt,tau)); air['A2_targeted'].append(float(mask.mean())); air['A2_random'].append(float(rmask.mean())); kdiff['A2'].append(int(mask.sum()-rmask.sum()))
        bout={}
        for name in rows:
            x=summarize_rows(rows[name]); x['AIR']=float(np.mean(air[name])); parent=stat if name.startswith('A1') else state; x['BDR_parent_detector']=float(np.mean(np.asarray(heldneg[parent])>= (th1 if name.startswith('A1') else th2))); bout[name]=x
        bout['exact_k_audit']={'A1_max_abs_k_diff':int(np.max(np.abs(kdiff['A1']))),'A2_max_abs_k_diff':int(np.max(np.abs(kdiff['A2']))),'pass':max(map(abs,kdiff['A1']+kdiff['A2']))==0}
        out['beta'][bk]=bout; out['_rows'][bk]=rows
    return out

def paired_effect(a,b):return float(a-b)

def utility_pair_effects(u):
    out={}
    for beta in PCFG['beta_grid']:
        b=str(beta); out[b]={}
        for attack in ('A1','A2'):
            t=u['beta'][b][attack+'_targeted']; r=u['beta'][b][attack+'_random']; out[b][attack]={k:paired_effect(t[k],r[k]) for k in ['CSVR','FSVR','Channel_Risk','Rejected']}; out[b][attack]['targeted_AIR']=t['AIR']; out[b][attack]['random_AIR']=r['AIR']; out[b][attack]['AIR_abs_diff']=abs(t['AIR']-r['AIR'])
    return out

def protocol_hashes(ret):
    files={}
    for p in sorted((PKG/'authorities').iterdir()):
        if p.is_file():files[p.name]=sha_file(p)
    files['g4_config.json']=sha_file(PKG/'config/g4_config.json'); files['project.json']=sha_file(PKG/'config/project.json')
    dump(ret,'G4_PROTOCOL_HASHES.json',{'files':files,'package_active_code':{p.name:sha_file(p) for p in sorted((PKG/'scripts').glob('*.py'))}})

def write_csvs(ret,cover,frontiers,utility,criteria):
    with open(Path(ret)/'detector_frontiers.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['condition','L','AIR','BDR','threshold']); w.writeheader()
        for cond in ALL_CONDITIONS:
            for L,rows in frontiers[cond].items():
                for r in rows:w.writerow({'condition':cond,'L':L,**r})
    with open(Path(ret)/'verdict_composition.csv','w',newline='',encoding='utf-8') as f:
        fields=['condition','beta','scenario','verdict','count','rate']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for cond in ALL_CONDITIONS:
            for v,x in utility[cond]['honest']['composition'].items():w.writerow({'condition':cond,'beta':0,'scenario':'Honest','verdict':v,**x})
            for beta in PCFG['beta_grid']:
                for sc in ['A1_targeted','A1_random','A2_targeted','A2_random']:
                    for v,x in utility[cond]['beta'][str(beta)][sc]['composition'].items():w.writerow({'condition':cond,'beta':beta,'scenario':sc,'verdict':v,**x})
    with open(Path(ret)/'pareto_table.csv','w',newline='',encoding='utf-8') as f:
        fields=['lambda','beta','L4_AUROC','L4_BDR_at_AIR_0.8','L4_AIR_at_BDR_0.1','honest_CSVR','A1_CSVR','A2_CSVR','A1_FSVR','A2_FSVR','A1_Channel_Risk','A2_Channel_Risk','A1_Rejected','A2_Rejected','C1_pass','C2_pass','C3_pass','C1_C2_C3_same_point_pass']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for lam in FULL_LAMBDAS:
            cond=f'M3_Full_{lam:g}'; c=criteria[str(lam)]
            for beta in PCFG['beta_grid']:
                b=str(beta); a1=utility[cond]['beta'][b]['A1_targeted']; a2=utility[cond]['beta'][b]['A2_targeted']; m=cover[cond]['4']; cp=c['per_beta'][b]
                w.writerow({'lambda':lam,'beta':beta,'L4_AUROC':m['AUROC_sym'],'L4_BDR_at_AIR_0.8':m['BDR_at_AIR_0.8'],'L4_AIR_at_BDR_0.1':m['AIR_at_BDR_0.1'],'honest_CSVR':utility[cond]['honest']['CSVR'],'A1_CSVR':a1['CSVR'],'A2_CSVR':a2['CSVR'],'A1_FSVR':a1['FSVR'],'A2_FSVR':a2['FSVR'],'A1_Channel_Risk':a1['Channel_Risk'],'A2_Channel_Risk':a2['Channel_Risk'],'A1_Rejected':a1['Rejected'],'A2_Rejected':a2['Rejected'],'C1_pass':c['C1']['pass'],'C2_pass':c['C2']['pass'],'C3_pass':cp['C3']['pass'],'C1_C2_C3_same_point_pass':cp['strong_point']})
    with open(Path(ret)/'fig_operating_region_source.csv','w',newline='',encoding='utf-8') as f:
        fields=['lambda','beta','BDR_gain_vs_unwrapped_L4','honest_CSVR','A1_CSVR','A2_CSVR','C1','C2','C3']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for lam in FULL_LAMBDAS:
            c=criteria[str(lam)]; cond=f'M3_Full_{lam:g}'
            for beta in PCFG['beta_grid']:
                cp=c['per_beta'][str(beta)]; w.writerow({'lambda':lam,'beta':beta,'BDR_gain_vs_unwrapped_L4':c['C1']['value'],'honest_CSVR':utility[cond]['honest']['CSVR'],'A1_CSVR':utility[cond]['beta'][str(beta)]['A1_targeted']['CSVR'],'A2_CSVR':utility[cond]['beta'][str(beta)]['A2_targeted']['CSVR'],'C1':c['C1']['pass'],'C2':c['C2']['pass'],'C3':cp['C3']['pass']})
    with open(Path(ret)/'fig_simple_baselines_source.csv','w',newline='',encoding='utf-8') as f:
        fields=['lambda','comparison','Full_BDR_L4','baseline_BDR_L4','delta_BDR_L4']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for lam in FULL_LAMBDAS:
            fc=cover[f'M3_Full_{lam:g}']['4']
            for base in BASE_METHODS:
                bc=cover[base]['4']; w.writerow({'lambda':lam,'comparison':f'Full-{base}','Full_BDR_L4':fc['BDR_at_AIR_0.8'],'baseline_BDR_L4':bc['BDR_at_AIR_0.8'],'delta_BDR_L4':fc['BDR_at_AIR_0.8']-bc['BDR_at_AIR_0.8']})

def build_criteria(cover,utility):
    out={}; uw=utility['M0_Unwrapped']; uw_cover=cover['M0_Unwrapped']['4']
    for lam in FULL_LAMBDAS:
        cond=f'M3_Full_{lam:g}'; u=utility[cond]; cv=cover[cond]['4']; c1v=cv['BDR_at_AIR_0.8']-uw_cover['BDR_at_AIR_0.8']; c2v=u['honest']['CSVR']-uw['honest']['CSVR']
        rec={'C1':{'definition':'L4 BDR@AIR=.8 Full-Unwrapped >= +0.20','value':c1v,'pass':c1v>=.20},'C2':{'definition':'Full honest CSVR drop vs Unwrapped <= 0.05','value':c2v,'pass':c2v>=-.05},'per_beta':{}}
        for beta in PCFG['beta_grid']:
            b=str(beta); a1=u['beta'][b]['A1_targeted']; a2=u['beta'][b]['A2_targeted']; u1=uw['beta'][b]['A1_targeted']; u2=uw['beta'][b]['A2_targeted']; hon=max(u['honest']['CSVR'],1e-12)
            x1={'gain':a1['CSVR']-u1['CSVR'],'retention':a1['CSVR']/hon}; x2={'gain':a2['CSVR']-u2['CSVR'],'retention':a2['CSVR']/hon}; c3=bool(x1['gain']>=.15 and x1['retention']>=.75 and x2['gain']>=.15 and x2['retention']>=.75)
            rec['per_beta'][b]={'C3':{'definition':'A1 AND A2 Full CSVR gain >= +0.15 AND retention >=75% Full honest','A1':x1,'A2':x2,'pass':c3},'strong_point':bool(rec['C1']['pass'] and rec['C2']['pass'] and c3)}
        out[str(lam)]=rec
    return out

def effect_summary(cover):
    out={}
    for lam in FULL_LAMBDAS:
        cond=f'M3_Full_{lam:g}'; out[str(lam)]={}
        for base in BASE_METHODS:
            out[str(lam)][base]={}
            for L in PCFG['L_grid']:
                a=cover[cond][str(L)]; b=cover[base][str(L)]; out[str(lam)][base][str(L)]={'delta_AUROC':a['AUROC_sym']-b['AUROC_sym'],'delta_BDR_at_AIR_0.8':a['BDR_at_AIR_0.8']-b['BDR_at_AIR_0.8'],'delta_AIR_at_BDR_0.1':a['AIR_at_BDR_0.1']-b['AIR_at_BDR_0.1']}
    return out

def bootstrap_cover(pool,J,state_marginal,sched,E,bundles,freezes,held_real,reps,logfn):
    by=collections.defaultdict(list)
    for i,r in enumerate(pool):by[r['conversation_id']].append(i)
    cids=sorted(by); rng=np.random.default_rng(PCFG['seed']+404040); negblocks={}; common=None
    for tag,bundle in bundles.items():
        bb,cc=heldout_neg_blocks(bundle,held_real,pool,'BOOTC_'+tag); negblocks[tag]=bb; common=set(cc) if common is None else common&set(cc)
    ncids=sorted(common or [])
    if not ncids:raise RuntimeError('no common fresh benign conversation blocks for G4 cover bootstrap')
    vals={}
    for lam in FULL_LAMBDAS:
        for base in BASE_METHODS:
            for L in PCFG['L_grid']:
                for met in ['AUROC','BDR08','AIR01']:vals[f'{lam:g}|{base}|L{L}|{met}']=[]
    for b in range(reps):
        pick=rng.integers(0,len(cids),len(cids)); bp=[]; orig=[]
        for copy_no,k in enumerate(pick):
            cid=cids[int(k)]
            for oi in by[cid]:
                r=dict(pool[oi]); r['id']=f"{r['id']}__b{b}c{copy_no}"; r['conversation_id']=f"{r['conversation_id']}__boot{copy_no}"; bp.append(r); orig.append(oi)
        if len(bp)<PCFG['main_q']:continue
        npick=rng.integers(0,len(ncids),len(ncids)); sampled=[ncids[int(i)] for i in npick]; neg={tag:sample_neg_blocks(negblocks[tag],sampled) for tag in bundles}
        needed=['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']
        if any(any(len(neg[tag][k])<2 for k in needed) for tag in bundles):continue
        orig=np.asarray(orig,int); Jb=np.asarray(J)[orig]; states=sched['state_fn']([r['text'] for r in bp]); sessions={}
        sessions['M0_Unwrapped'],_=condition_sessions('M0_Unwrapped',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTC_{b}')
        sessions['M1_MarginalCover'],_=condition_sessions('M1_MarginalCover',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTC_{b}')
        sessions['M2_ScheduleOnly'],_=condition_sessions('M2_ScheduleOnly',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTC_{b}')
        for lam in FULL_LAMBDAS:sessions[f'M3_Full_{lam:g}'],_=condition_sessions('M3_Full',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTC_{b}',lam)
        mets={}
        for cond in ALL_CONDITIONS:
            tag='M3' if cond.startswith('M3_') else cond.split('_')[0]; mets[cond],_=metric_by_freeze(bundles[tag],bp,sessions[cond],neg[tag],freezes[cond])
        for lam in FULL_LAMBDAS:
            fc=mets[f'M3_Full_{lam:g}']
            for base in BASE_METHODS:
                for L in PCFG['L_grid']:
                    a=fc[str(L)]; z=mets[base][str(L)]; vals[f'{lam:g}|{base}|L{L}|AUROC'].append(a['AUROC_sym']-z['AUROC_sym']); vals[f'{lam:g}|{base}|L{L}|BDR08'].append(a['BDR_at_AIR_0.8']-z['BDR_at_AIR_0.8']); vals[f'{lam:g}|{base}|L{L}|AIR01'].append(a['AIR_at_BDR_0.1']-z['AIR_at_BDR_0.1'])
        if (b+1)%50==0:logfn(f'G4 cover cluster bootstrap {b+1}/{reps}')
    out={}
    for k,a in vals.items():
        x=np.asarray(a,float); out[k]={'n':int(len(x)),'mean':float(np.mean(x)) if len(x) else None,'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))] if len(x) else [None,None]}
    completed=min((len(v) for v in vals.values()),default=0)
    if completed<int(reps*.95):raise RuntimeError(f'cover cluster bootstrap valid replicates too few: {completed}/{reps}')
    return {'procedure':'paired WildChat conversation-cluster bootstrap; query clusters resampled and all M0/M1/M2/M3(lambda) sessions reconstructed with shared frozen session seeds; fresh benign detector blocks resampled by original conversation','replicates_requested':reps,'replicates_completed':completed,'effects':out,'naive_session_bootstrap_primary':False}

def bootstrap_random_utility(pool,J,state_marginal,sched,E,bundles,freezes,lin,reps,logfn):
    # Matched cluster reconstruction for the targeted-vs-exact-k-random diagnostic.
    by=collections.defaultdict(list)
    for i,r in enumerate(pool):by[r['conversation_id']].append(i)
    cids=sorted(by); rng=np.random.default_rng(PCFG['seed']+505050); vals={}
    for cond in ALL_CONDITIONS:
        for beta in PCFG['beta_grid']:
            for attack in ['A1','A2']:
                for met in ['CSVR','FSVR','Channel_Risk','Rejected']:vals[f'{cond}|{beta}|{attack}|{met}']=[]
    for b in range(reps):
        pick=rng.integers(0,len(cids),len(cids)); bp=[]; orig=[]
        for copy_no,k in enumerate(pick):
            cid=cids[int(k)]
            for oi in by[cid]:
                r=dict(pool[oi]); r['id']=f"{r['id']}__u{b}c{copy_no}"; r['conversation_id']=f"{r['conversation_id']}__boot{copy_no}"; bp.append(r); orig.append(oi)
        if len(bp)<PCFG['main_q']:continue
        orig=np.asarray(orig,int); Eb={k:np.asarray(v)[orig] for k,v in E.items()}; Jb=np.asarray(J)[orig]; states=sched['state_fn']([r['text'] for r in bp]); sessions={}
        sessions['M0_Unwrapped'],_=condition_sessions('M0_Unwrapped',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTU_{b}')
        sessions['M1_MarginalCover'],_=condition_sessions('M1_MarginalCover',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTU_{b}')
        sessions['M2_ScheduleOnly'],_=condition_sessions('M2_ScheduleOnly',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTU_{b}')
        for lam in FULL_LAMBDAS:sessions[f'M3_Full_{lam:g}'],_=condition_sessions('M3_Full',bp,Jb,states,sched,state_marginal,PCFG['heldout_sessions'],PCFG['main_q'],f'G4_BOOTU_{b}',lam)
        for cond in ALL_CONDITIONS:
            kind='M3_Full' if cond.startswith('M3_') else cond; tag='M3' if cond.startswith('M3_') else cond.split('_')[0]; bundle=bundles[tag]; freeze=freezes[cond]; ad=C.DNAAdapter(); rt=lin['risk_threshold_kappa'] if riskaware(kind) else None; gt=lin['bank_gap_threshold'] if riskaware(kind) else None; tau=lin['similarity_threshold']
            for beta in PCFG['beta_grid']:
                bk=str(beta); stat=freeze['stateless_attacker']; state=freeze['stateful_attacker_L4']; th1=freeze['thresholds'][bk][stat]['threshold']; th2=freeze['thresholds'][bk][state]['threshold']; rows={x:[] for x in ['A1t','A1r','A2t','A2r']}
                for s in sessions[cond]:
                    eo=evidence_order(kind,s); mix,mask,_=a1_target(s,bp,bundle,stat,th1,Eb['qwen_nf4'],Eb['phi_source']); rmix,_=random_exact_k(s,mask,Eb['qwen_nf4'],Eb['phi_source'],random_seed(cond,beta,s['episode'],'A1'),0); rows['A1t'].append(C.lineage_session(ad,eo,mix,Eb['qwen_source'],Eb['phi_source'],rt,gt,tau)); rows['A1r'].append(C.lineage_session(ad,eo,rmix,Eb['qwen_source'],Eb['phi_source'],rt,gt,tau)); mix,mask,_=a2_target(s,bp,bundle,state,th2,Eb['qwen_nf4'],Eb['phi_source'],4); rmix,_=random_exact_k(s,mask,Eb['qwen_nf4'],Eb['phi_source'],random_seed(cond,beta,s['episode'],'A2'),3); rows['A2t'].append(C.lineage_session(ad,eo,mix,Eb['qwen_source'],Eb['phi_source'],rt,gt,tau)); rows['A2r'].append(C.lineage_session(ad,eo,rmix,Eb['qwen_source'],Eb['phi_source'],rt,gt,tau))
                for attack,t,r in [('A1','A1t','A1r'),('A2','A2t','A2r')]:
                    T=C.verdict_rates(rows[t]); R=C.verdict_rates(rows[r])
                    for met in ['CSVR','FSVR','Channel_Risk','Rejected']:vals[f'{cond}|{beta}|{attack}|{met}'].append(float(T[met]-R[met]))
        if (b+1)%25==0:logfn(f'G4 random-routing cluster bootstrap {b+1}/{reps}')
    out={}
    for k,a in vals.items():
        x=np.asarray(a,float); out[k]={'n':int(len(x)),'mean':float(np.mean(x)) if len(x) else None,'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))] if len(x) else [None,None]}
    completed=min((len(v) for v in vals.values()),default=0)
    if completed<int(reps*.95):raise RuntimeError(f'random-routing cluster bootstrap valid replicates too few: {completed}/{reps}')
    return {'procedure':'literal paired WildChat conversation-cluster bootstrap; underlying G4 query clusters resampled and all matched method sessions reconstructed in each replicate; targeted and exact-k random counterfactuals share parent session and frozen RNG key','replicates_requested':reps,'replicates_completed':completed,'effects':out,'naive_session_bootstrap_primary':False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); args=ap.parse_args(); rd=Path(args.run_dir); ret=rd/'return'; ret.mkdir(parents=True,exist_ok=True); configure_core(rd)
    heldout_touched=False; deviations=[]; phases={}
    try:
        pre=load_json(rd/'precheck/precheck_report.json')
        if not pre or pre.get('status')!='PASS':raise RuntimeError('precheck PASS required before G4 execution')
        protocol_hashes(ret); shutil.copy2(rd/'precheck/method_definitions.json',ret/'method_definitions.json'); shutil.copy2(rd/'precheck/external_baseline_feasibility.json',ret/'external_baseline_feasibility.json')
        update_status(rd,status='RUNNING',phase='DEVCAL_REFREEZE',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY'); log(rd,'=== G4 operating-region + simple-baseline dev/cal freeze ===')
        dev_real,cal_real,dev_pool,cal_pool=reconstructed_devcal(); rec=[{**r,'pool':'dev'} for r in dev_pool]+[{**r,'pool':'cal'} for r in cal_pool]; banks,reuse=acquire_devcal_banks(rec); Eall=embed_banks(banks); nd=len(dev_pool); devE={k:v[:nd] for k,v in Eall.items()}; calE={k:v[nd:] for k,v in Eall.items()}; phases['devcal_banks']='ok'
        Jd,scale,_=source_only_J(devE['qwen_source'],devE['phi_source']); Jc,_,_=source_only_J(calE['qwen_source'],calE['phi_source'],scale); info={'definition':'cosdist(Qwen-source,Phi-source), first64; development q05/q95 scaling; no NF4/descendant/suspect information in selection','first_coordinates':64,'development_scaling':{'q05':scale[0],'q95':scale[1]},'forbidden_inputs':['NF4','LoRA','suspect/descendant responses','descendant-derived statistics'],'asset_reuse':reuse,'self_audit':'PASS'}
        from sentence_transformers import SentenceTransformer
        mp=SentenceTransformer(str(C.MPNET),device='cuda:0'); sched=C.build_scheduler([{'id':s['id'],'texts':s['texts']} for s in dev_real],dev_pool,cal_pool,mp); marginal,marginal_meta=state_marginal_from_dev(sched,dev_real); phases['scheduler_marginal']='ok'
        # Frozen dev/cal sessions. Same paired seed set per episode across all conditions where applicable.
        dev_sessions={}; cal_sessions={}
        dev_sessions['M0_Unwrapped'],_=condition_sessions('M0_Unwrapped',dev_pool,Jd,sched['states_dev'],sched,marginal,PCFG['dev_sessions_per_condition'],64,'G4_DEV')
        dev_sessions['M1_MarginalCover'],_=condition_sessions('M1_MarginalCover',dev_pool,Jd,sched['states_dev'],sched,marginal,PCFG['dev_sessions_per_condition'],64,'G4_DEV')
        dev_sessions['M2_ScheduleOnly'],_=condition_sessions('M2_ScheduleOnly',dev_pool,Jd,sched['states_dev'],sched,marginal,PCFG['dev_sessions_per_condition'],64,'G4_DEV')
        cal_sessions['M0_Unwrapped'],_=condition_sessions('M0_Unwrapped',cal_pool,Jc,sched['states_cal'],sched,marginal,PCFG['cal_sessions_per_condition'],64,'G4_CAL')
        cal_sessions['M1_MarginalCover'],_=condition_sessions('M1_MarginalCover',cal_pool,Jc,sched['states_cal'],sched,marginal,PCFG['cal_sessions_per_condition'],64,'G4_CAL')
        cal_sessions['M2_ScheduleOnly'],_=condition_sessions('M2_ScheduleOnly',cal_pool,Jc,sched['states_cal'],sched,marginal,PCFG['cal_sessions_per_condition'],64,'G4_CAL')
        for lam in FULL_LAMBDAS:
            dev_sessions[f'M3_Full_{lam:g}'],_=condition_sessions('M3_Full',dev_pool,Jd,sched['states_dev'],sched,marginal,PCFG['dev_sessions_per_condition'],64,'G4_DEV',lam); cal_sessions[f'M3_Full_{lam:g}'],_=condition_sessions('M3_Full',cal_pool,Jc,sched['states_cal'],sched,marginal,PCFG['cal_sessions_per_condition'],64,'G4_CAL',lam)
        bundles={'M0':train_bundle('M0',dev_real,dev_pool,[dev_sessions['M0_Unwrapped']],mp),'M1':train_bundle('M1',dev_real,dev_pool,[dev_sessions['M1_MarginalCover']],mp),'M2':train_bundle('M2',dev_real,dev_pool,[dev_sessions['M2_ScheduleOnly']],mp),'M3':train_bundle('M3',dev_real,dev_pool,[dev_sessions[f'M3_Full_{l:g}'] for l in FULL_LAMBDAS],mp)}; phases['detectors']='ok'
        calneg={k:calibration_neg(v,cal_real,cal_pool,k) for k,v in bundles.items()}; base_cal,_=condition_sessions('M0_Unwrapped',cal_pool,Jc,sched['states_cal'],sched,marginal,PCFG['lineage_cal_sessions'],64,'G4_LINEAGE_CAL'); lin=calibrate_lineage(base_cal,calE)
        freezes={'M0_Unwrapped':detector_freeze(bundles['M0'],cal_pool,cal_sessions['M0_Unwrapped'],calneg['M0']),'M1_MarginalCover':detector_freeze(bundles['M1'],cal_pool,cal_sessions['M1_MarginalCover'],calneg['M1']),'M2_ScheduleOnly':detector_freeze(bundles['M2'],cal_pool,cal_sessions['M2_ScheduleOnly'],calneg['M2'])}
        for lam in FULL_LAMBDAS:freezes[f'M3_Full_{lam:g}']=detector_freeze(bundles['M3'],cal_pool,cal_sessions[f'M3_Full_{lam:g}'],calneg['M3'])
        elig=load_json(rd/'precheck/eligibility.json'); pool=load_pool_jsonl(rd/'precheck/eligible_pool.jsonl'); freeze_manifest={'stage':'G4_FREEZE_BEFORE_HELDOUT_RESPONSE_ACCESS','frozen_at':now(),'heldout_response_access':False,'heldout_metrics_access':False,'fresh_pool_n':len(pool),'fresh_pool_ids_sha256':elig['hashes']['chosen_ids_sha256'],'fresh_pool_prior_overlap_conversations':elig['prior_overlap_conversations'],'fresh_pool_prior_overlap_texts':elig['prior_overlap_texts'],'q':64,'session_realizations':64,'lambda_grid':list(FULL_LAMBDAS),'lambda_selection':'FORBIDDEN_ALL_FIVE_REPORTED','beta_grid':PCFG['beta_grid'],'source_only_scorer':info,'benign_state_marginal':marginal_meta,'method_detector_freezes':freezes,'lineage_thresholds':lin,'random_routing_rng_rule':'SHA256-derived deterministic seed keyed by condition|beta|session|attack; exact targeted k matched per parent session','bootstrap_reps':2000,'external_baseline_feasibility_timing':'PRE_HELDOUT','downstream_stage_invocation':False,'retuning_after_heldout':False}; dump(ret,'g4_freeze_manifest.json',freeze_manifest); phases['freeze']='ok'
        heldout_touched=True; update_status(rd,status='RUNNING',phase='HELDOUT_ONE_SHOT',heldout_touched=True,scientific_state_advanced=True,resume_allowed='NO'); log(rd,f'G4 HELDOUT ACCESS AUTHORIZED ONCE: fresh N={len(pool)}; lambdas={list(FULL_LAMBDAS)}; betas={PCFG["beta_grid"]}')
        held_real=fresh_held_real(pool); hbanks={}
        for tag,path,kind in [('g4_held_qwen_source',C.QWEN,'source'),('g4_held_qwen_nf4',C.QWEN,'nf4'),('g4_held_phi_source',C.PHI,'phi')]:hbanks[tag]=C.generate_bank(tag,path,kind,pool)
        H0=embed_banks(hbanks); HE={'qwen_source':H0['g4_held_qwen_source'],'qwen_nf4':H0['g4_held_qwen_nf4'],'phi_source':H0['g4_held_phi_source']}; Jh,_,_=source_only_J(HE['qwen_source'],HE['phi_source'],scale); states=sched['state_fn']([r['text'] for r in pool]); phases['heldout_bank']='ok'
        sessions={}; sessions['M0_Unwrapped'],_=condition_sessions('M0_Unwrapped',pool,Jh,states,sched,marginal,64,64,'G4_HELD'); sessions['M1_MarginalCover'],_=condition_sessions('M1_MarginalCover',pool,Jh,states,sched,marginal,64,64,'G4_HELD'); sessions['M2_ScheduleOnly'],_=condition_sessions('M2_ScheduleOnly',pool,Jh,states,sched,marginal,64,64,'G4_HELD')
        for lam in FULL_LAMBDAS:sessions[f'M3_Full_{lam:g}'],_=condition_sessions('M3_Full',pool,Jh,states,sched,marginal,64,64,'G4_HELD',lam)
        manifests=[]
        for cond in ALL_CONDITIONS:
            kind='M3_Full' if cond.startswith('M3_') else cond; lam=float(cond.split('_')[-1]) if cond.startswith('M3_') else None; manifests+=session_manifest_rows(cond,pool,sessions[cond],Jh,states,marginal,kind,lam)
        jsonl_write(ret/'g4_session_manifest.jsonl',manifests); dump(ret,'g4_heldout_pool_manifest.json',{'n':len(pool),'minimum_required':CFG['candidate_min'],'target':CFG['candidate_target'],'q':64,'query_ids_sha256':elig['hashes']['chosen_ids_sha256'],'conversation_ids_sha256':elig['hashes']['chosen_conversation_ids_sha256'],'text_hashes_sha256':elig['hashes']['chosen_text_hashes_sha256'],'prior_overlap_conversations':elig['prior_overlap_conversations'],'prior_overlap_texts':elig['prior_overlap_texts'],'heldout_response_access':True,'eligibility':elig})
        dump(ret,'g4_overlap_diagnostics.json',{cond:overlap_diag(pool,sessions[cond],Jh) for cond in ALL_CONDITIONS})
        heldneg={k:heldout_neg_scores(v,held_real,pool,k) for k,v in bundles.items()}; cover={}; frontiers={}
        for cond in ALL_CONDITIONS:
            tag='M3' if cond.startswith('M3_') else cond.split('_')[0]; cover[cond],frontiers[cond]=metric_by_freeze(bundles[tag],pool,sessions[cond],heldneg[tag],freezes[cond])
            if cond.startswith('M3_'):cover[cond]['source_only_selection']=full_theory(Jh,sessions[cond],float(cond.split('_')[-1]))
        dump(ret,'cover_results.json',cover); phases['cover']='ok'
        utility={}; random_results={}
        for cond in ALL_CONDITIONS:
            kind='M3_Full' if cond.startswith('M3_') else cond; tag='M3' if cond.startswith('M3_') else cond.split('_')[0]; u=eval_utility_sweep(cond,kind,sessions[cond],pool,HE,bundles[tag],freezes[cond],lin,heldneg[tag]); random_results[cond]=utility_pair_effects(u); u.pop('_rows',None); utility[cond]=u; log(rd,f'utility sweep complete: {cond}')
        dump(ret,'utility_sweep_results.json',utility); dump(ret,'random_routing_results.json',random_results); effects=effect_summary(cover); criteria=build_criteria(cover,utility); phases['utility']='ok'
        # Frozen primary cluster-aware uncertainty. These routines do not tune or select any point.
        cover_ci=bootstrap_cover(pool,Jh,marginal,sched,HE,bundles,freezes,held_real,int(PCFG['bootstrap_reps']),lambda x:log(rd,x)); dump(ret,'cluster_bootstrap_cover_CI.json',cover_ci); phases['cover_bootstrap']='ok'
        random_ci=bootstrap_random_utility(pool,Jh,marginal,sched,HE,bundles,freezes,lin,int(PCFG['bootstrap_reps']),lambda x:log(rd,x)); phases['random_bootstrap']='ok'
        # Merge CIs into the canonical single required file and matched effects.
        dump(ret,'cluster_bootstrap_CI.json',{'cover_method_differences':cover_ci,'targeted_vs_AIR_matched_random':random_ci})
        for lam in FULL_LAMBDAS:
            for base in BASE_METHODS:
                for L in PCFG['L_grid']:
                    e=effects[str(lam)][base][str(L)]; e['ci95_delta_AUROC']=cover_ci['effects'][f'{lam:g}|{base}|L{L}|AUROC']['ci95']; e['ci95_delta_BDR_at_AIR_0.8']=cover_ci['effects'][f'{lam:g}|{base}|L{L}|BDR08']['ci95']; e['ci95_delta_AIR_at_BDR_0.1']=cover_ci['effects'][f'{lam:g}|{base}|L{L}|AIR01']['ci95']
        for cond in ALL_CONDITIONS:
            for beta in PCFG['beta_grid']:
                for attack in ['A1','A2']:
                    for met in ['CSVR','FSVR','Channel_Risk','Rejected']:
                        random_results[cond][str(beta)][attack][f'ci95_delta_{met}']=random_ci['effects'][f'{cond}|{beta}|{attack}|{met}']['ci95']
        dump(ret,'matched_method_effects.json',effects); dump(ret,'random_routing_results.json',random_results); write_csvs(ret,cover,frontiers,utility,criteria)
        # Mechanical criteria only. paper analysis owns Axis A/B adjudication.
        mechanical={'criteria_by_lambda':criteria,'any_C1_C2_C3_same_point':bool(any(v['strong_point'] for c in criteria.values() for v in c['per_beta'].values())),'method_specificity_ingredients':{}}
        for lam in FULL_LAMBDAS:
            x={}
            for base in ['M1_MarginalCover','M2_ScheduleOnly']:
                e=effects[str(lam)][base]['4']; x[base]={'delta_BDR':e['delta_BDR_at_AIR_0.8'],'ci95':e['ci95_delta_BDR_at_AIR_0.8'],'meets_plus_0.20_and_ci_lower_gt_0':bool(e['delta_BDR_at_AIR_0.8']>=.20 and e['ci95_delta_BDR_at_AIR_0.8'][0]>0)}
            mechanical['method_specificity_ingredients'][str(lam)]=x
        dump(ret,'mechanical_criteria_checks.json',mechanical)
        with open(ret/'EXECUTION_DEVIATIONS.md','w',encoding='utf-8') as f:f.write('# EXECUTION DEVIATIONS\n\n'+('None. Frozen G4 protocol executed without scientific deviation.\n' if not deviations else '\n'.join('- '+x for x in deviations)+'\n'))
        with open(ret/'paper_result_summary.md','w',encoding='utf-8') as f:
            f.write('# SLATE Operating Region + Simple Baselines — Execution Result\n\n'); f.write(f'- fresh held-out N: {len(pool)}; q=64; sessions per condition=64\n- prior-held-out overlap conversations: {elig["prior_overlap_conversations"]}; prior-held-out overlap texts: {elig["prior_overlap_texts"]}\n- held-out touched: YES (one-shot after freeze)\n- deviations: {"NO" if not deviations else "YES"}\n- lambda grid: {list(FULL_LAMBDAS)} (ALL reported; no best-lambda selection)\n- beta grid: {PCFG["beta_grid"]}\n- downstream experiment expansion executed: NO\n- retuning/rescue/second pool: NO\n\n'); f.write('## Mechanical C1/C2/C3 checks (not Axis-A adjudication)\n```json\n'+json.dumps(criteria,indent=2)+'\n```\n\n'); f.write('## Method-comparison measured effects with cluster CIs (not Axis-B adjudication)\n```json\n'+json.dumps(effects,indent=2)+'\n```\n\n'); f.write('## Targeted minus AIR-matched-random paired effects\n```json\n'+json.dumps(random_results,indent=2)+'\n```\n\n'); f.write('execution layer reports facts and frozen criterion checks only. This runner reports facts and frozen criterion checks only; scientific interpretation is performed separately.\n')
        exact=ret/'exact_code_config'; shutil.copytree(PKG/'scripts',exact/'scripts',dirs_exist_ok=True); shutil.copytree(PKG/'config',exact/'config',dirs_exist_ok=True); rows=tree_manifest(ret,exclude=('MANIFEST_SHA256.txt',));
        with open(ret/'MANIFEST_SHA256.txt','w',encoding='utf-8') as f:
            for r in rows:f.write(f"{r['sha256']}  {r['path']}\n")
        update_status(rd,status='SUCCESS',phase='COMPLETE',heldout_touched=True,scientific_state_advanced=True,resume_allowed='NO',exit_code=0); log(rd,'G4 EXECUTION SUCCESS: evidence complete; scientific adjudication deferred to paper analysis'); return 0
    except Exception as e:
        dump(ret,'EXECUTION_ERROR.json',{'error':repr(e),'traceback':traceback.format_exc(),'heldout_touched':heldout_touched,'phases':phases,'time':now()}); (ret/'EXECUTION_DEVIATIONS.md').write_text('# EXECUTION DEVIATIONS\n\nExecution failure; no scientific substitution, retuning, second pool, or rescue was attempted.\n',encoding='utf-8'); (ret/'paper_result_summary.md').write_text(f'# SLATE G4 Execution Failure\n\n- held-out touched: {"YES" if heldout_touched else "NO"}\n- error: `{repr(e)}`\n- retuning/rescue/second pool: NO\n- downstream experiment expansion executed: NO\n- scientific adjudication: NOT PERFORMED\n',encoding='utf-8'); update_status(rd,status='FAILED',phase='HELDOUT_FAILURE' if heldout_touched else 'PREHELDOUT_FAILURE',heldout_touched=heldout_touched,scientific_state_advanced=heldout_touched,resume_allowed='NO' if heldout_touched else 'CONDITIONAL_MECHANICAL_ONLY',failure_summary=repr(e),exit_code=1); log(rd,traceback.format_exc()); return 1

if __name__=='__main__':sys.exit(main())
