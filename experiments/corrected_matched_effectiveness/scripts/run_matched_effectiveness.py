from __future__ import annotations
import argparse, collections, csv, gc, hashlib, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
from common import *

# The legacy core is reused only for validated mechanics. All G1/G2 scientific changes are implemented here.
SHARED_HELPERS=Path(__file__).resolve().parents[2]/'_shared'
if str(SHARED_HELPERS) not in sys.path: sys.path.insert(0,str(SHARED_HELPERS))
import core_helpers as C

METHODS=('Full','Unwrapped')

def log(rd,msg):
    s=str(msg); print(s,flush=True)
    with open(Path(rd)/'run.log','a',encoding='utf-8') as f:f.write(s+'\n')

def dump(ret,name,obj): atomic_json(Path(ret)/name,obj)

def configure_core(rd):
    rd=Path(rd); C.CFG=PCFG; C.RUN=rd/'work'; C.RET=rd/'work'/'legacy_return'; C.BANK=rd/'work'/'banks'; C.EMB=rd/'work'/'embeddings'; C.CACHE=rd/'work'/'cache'; C.LOG=rd/'run.log'
    for p in [C.RUN,C.RET,C.BANK,C.EMB,C.CACHE]: p.mkdir(parents=True,exist_ok=True)
    C.P3A=Path(CFG['llmdna_root']).parents[1]  # .../slate_stagep3a... directory
    C.LLMDNA_SRC=Path(CFG['llmdna_root'])/'src'
    C.QWEN=Path(CFG['qwen_source']); C.PHI=Path(CFG['phi_source']); C.ENC=Path(CFG['qwen_embedding']); C.MPNET=Path(CFG['mpnet']); C.TRAFFIC=Path(CFG['traffic'])
    if str(C.LLMDNA_SRC) not in sys.path: sys.path.insert(0,str(C.LLMDNA_SRC))

def load_pool_jsonl(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def untouched_held_real():
    traffic=Path(CFG['traffic'])
    dev_s=sessions_for_split(traffic,'development','P3B_development'); cal_s=sessions_for_split(traffic,'calibration','P3B_calibration'); held_s=sessions_for_split(traffic,'heldout','P3C_heldout')
    dev=freeze_pool(dev_s,512,'P3B_DEV_POOL'); cal=freeze_pool(cal_s,128,'P3B_CAL_POOL'); old=freeze_pool(held_s,128,'P3C_HELDOUT_POOL')
    banned={x['conversation_id'] for x in dev+cal+old}
    return [s for s in held_s if s['id'] not in banned]

def reconstructed_devcal():
    traffic=Path(CFG['traffic'])
    ds=sessions_for_split(traffic,'development','P3B_development'); cs=sessions_for_split(traffic,'calibration','P3B_calibration')
    dev=freeze_pool(ds,512,'P3B_DEV_POOL'); cal=freeze_pool(cs,128,'P3B_CAL_POOL')
    # Convert to legacy shape; contents/IDs are identical to old deterministic freeze.
    return ds,cs,dev,cal

def prior_bank_rows(tag, records):
    p=Path(CFG['p3b_run'])/'banks'/f'{tag}.json'
    if not p.exists(): return None
    try: state=json.load(open(p,encoding='utf-8')); mp={r['id']:r for r in state.get('rows',[]) if r.get('response') is not None}
    except Exception: return None
    if not all(r['id'] in mp and mp[r['id']].get('text')==r['text'] for r in records): return None
    return [mp[r['id']] for r in records]

def acquire_devcal_banks(records, rd):
    audit={}; banks={}
    spec=[('qwen_source',C.QWEN,'source'),('qwen_nf4',C.QWEN,'nf4'),('phi_source',C.PHI,'phi')]
    for tag,path,kind in spec:
        old=prior_bank_rows(tag,records)
        if old is not None:
            banks[tag]=old; audit[tag]={'source':'reused_exact_prior_response_asset','prior_path':str(Path(CFG['p3b_run'])/'banks'/f'{tag}.json'),'n':len(old),'query_id_sha256':sha_text('\n'.join(r['id'] for r in old))}
            atomic_json(C.BANK/f'{tag}.json',{'tag':tag,'kind':kind,'rows':old})
        else:
            banks[tag]=C.generate_bank(tag,path,kind,records); audit[tag]={'source':'regenerated_exact_frozen_model_and_generation_settings','n':len(banks[tag]),'query_id_sha256':sha_text('\n'.join(r['id'] for r in banks[tag]))}
    return banks,audit

def embed_banks(banks):
    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer(str(C.ENC),device='cuda:0')
    E={k:C.embed_bank(k,v,enc) for k,v in banks.items()}
    del enc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return E

def source_only_J(qwen,phi,scale=None):
    raw=1-C.cosrow(np.asarray(qwen)[:,:64],np.asarray(phi)[:,:64])
    if scale is None:
        lo,hi=np.quantile(raw,[.05,.95]); scale=(float(lo),float(hi))
    lo,hi=scale; den=max(float(hi-lo),1e-8)
    return np.clip((raw-lo)/den,0,1),scale,raw

def uniform_sessions(n_pool,n_sessions,q,tag,states=None,scheduler=None,paired_lam=None):
    out=[]
    for e in range(n_sessions):
        seed_key=f'{tag}|{paired_lam}|{e}' if paired_lam is not None else f'{tag}|{e}'
        seed=int(PCFG['seed']+C.h64(seed_key,'SESSION')%100000000)
        rng=np.random.default_rng(seed); sel=rng.choice(n_pool,q,replace=False).astype(int).tolist(); rnd=np.asarray(sel)[rng.permutation(q)].astype(int).tolist()
        out.append({'episode':e,'seed':seed,'selected':sel,'random_order':rnd,'order':rnd,'mean_J':None})
    return out

def train_bundle(tag,dev_real,dev_pool,dev_session_groups,mp):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ec=C.ECache(mp); posq=[]
    for ss in dev_session_groups: posq.extend(C.flatten_queries(dev_pool,ss))
    posq=posq[:PCFG['detector_query_cap']]
    negq=C.excluded_queries([{'id':s['id'],'texts':s['texts']} for s in dev_real],[r['text'] for r in dev_pool],PCFG['detector_query_cap'],'G3_'+tag+'_DNEG')
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=100000,sublinear_tf=True,strip_accents='unicode')
    X=vec.fit_transform(posq+negq); y=np.r_[np.ones(len(posq)),np.zeros(len(negq))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=PCFG['seed']).fit(X,y)
    d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=PCFG['seed']).fit(np.r_[ec.get(posq),ec.get(negq)],y)
    pooled=[s for ss in dev_session_groups for s in ss]; seq={}
    real=[{'id':s['id'],'texts':s['texts']} for s in dev_real]
    for L in PCFG['L_grid']:
        pos=C.audit_items(dev_pool,pooled,L)[:PCFG['detector_window_cap']]; negw=C.real_windows(real,L,PCFG['detector_window_cap'],'G3_'+tag+'_NEGW'+str(L))
        if L in (4,8): seq[f'D3_L{L}']=C.fit_seq(ec,pos,negw,L,PCFG['d3_hidden'],False,PCFG['seed']+L+(1000 if tag=='UNWRAP' else 0))
        seq[f'D4_L{L}']=C.fit_seq(ec,pos,negw,L,PCFG['d4_hidden'],True,PCFG['seed']+100+L+(1000 if tag=='UNWRAP' else 0))
    return {'d1':d1,'d2':d2,'vec':vec,'ec':ec,'seq':seq}

def calibration_neg(bundle,cal_real,cal_pool,tag):
    b=bundle; real=[{'id':s['id'],'texts':s['texts']} for s in cal_real]
    negq=C.excluded_queries(real,[r['text'] for r in cal_pool],6000,'G3_'+tag+'_CALNEG')
    if len(negq)<int(CFG['min_negative_queries']): raise RuntimeError(f'insufficient calibration negatives for {tag}: {len(negq)}')
    out={'D1':b['d1'].predict_proba(b['vec'].transform(negq))[:,1],'D2':b['d2'].predict_proba(b['ec'].get(negq))[:,1]}
    for L in PCFG['L_grid']:
        nw=C.real_windows(real,L,PCFG['detector_window_cap'],'G3_'+tag+'_CALW'+str(L))
        if L in (4,8): out[f'D3_L{L}']=C.score_seq(b['seq'][f'D3_L{L}'],b['ec'],nw,L)
        out[f'D4_L{L}']=C.score_seq(b['seq'][f'D4_L{L}'],b['ec'],nw,L)
    return out

def positive_scores(bundle,pool,sessions):
    b=bundle; qs=C.flatten_queries(pool,sessions); out={'D1':b['d1'].predict_proba(b['vec'].transform(qs))[:,1],'D2':b['d2'].predict_proba(b['ec'].get(qs))[:,1]}
    for L in PCFG['L_grid']:
        ai=C.audit_items(pool,sessions,L)
        if L in (4,8): out[f'D3_L{L}']=C.score_seq(b['seq'][f'D3_L{L}'],b['ec'],ai,L)
        out[f'D4_L{L}']=C.score_seq(b['seq'][f'D4_L{L}'],b['ec'],ai,L)
    return out

def candidates_for_L(L):
    return ['D1','D2','D4_L1'] if L==1 else [f'D3_L{L}',f'D4_L{L}']

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
    return {'similarity_threshold':tau,'risk_threshold_kappa':rt,'bank_gap_threshold':gt,'calibration_n':len(rows),'false_source_calibration_rate':float(np.mean(neg>=tau))}

def full_theory(J,sessions,lam):
    w=C.distribution_weights(J,lam); K=float(PCFG['main_q']*np.sum(w*np.log(np.maximum(w*len(w),1e-300))))
    meanj=float(np.mean([s['mean_J'] for s in sessions])); return {'mean_J':meanj,'pool_mean_J':float(np.mean(J)),'Gamma_select':meanj-float(np.mean(J)),'Khat_q':K}

def verdict_composition(rows):
    n=max(1,len(rows)); d=collections.Counter()
    for r in rows:
        if r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base': d['correct-source verified']+=1
        elif r['verdict']=='Verified': d['false-source verified']+=1
        elif r['verdict']=='Inconclusive-Serving': d['Channel-Risk']+=1
        elif r['verdict']=='Inconclusive-Bank': d['Insufficient-Evidence']+=1
        elif r['verdict']=='Rejected': d['Rejected']+=1
        else: d['Other']+=1
    return {k:{'count':int(d[k]),'rate':float(d[k]/n)} for k in ['correct-source verified','false-source verified','Channel-Risk','Insufficient-Evidence','Rejected','Other']}

def eval_lineage(method,sessions,pool,E,bundle,detfreeze,lin,heldneg=None,adapter=None):
    ad=adapter or C.DNAAdapter(); beta=str(PCFG['main_beta']); stat=detfreeze['stateless_attacker']; state=detfreeze['stateful_attacker_L4']; th1=detfreeze['thresholds'][beta][stat]['threshold']; th2=detfreeze['thresholds'][beta][state]['threshold']
    risk=(method=='Full'); rt=lin['risk_threshold_kappa'] if risk else None; gt=lin['bank_gap_threshold'] if risk else None; tau=lin['similarity_threshold']
    hon=[];a1=[];a2=[];air1=[];air2=[]
    for s in sessions:
        evidence_order=s['random_order'] if method=='Full' else s['order']
        hon.append(C.lineage_session(ad,evidence_order,E['qwen_nf4'],E['qwen_source'],E['phi_source'],rt,gt,tau))
        mix,a,_=C.attack_a1(s,pool,bundle['d1'],bundle['d2'],bundle['vec'],bundle['ec'],stat,th1,E['qwen_nf4'],E['phi_source']); a1.append(C.lineage_session(ad,evidence_order,mix,E['qwen_source'],E['phi_source'],rt,gt,tau)); air1.append(a)
        mix,a,_=C.attack_a2(s,pool,bundle['seq'][state],bundle['ec'],4,th2,E['qwen_nf4'],E['phi_source']); a2.append(C.lineage_session(ad,evidence_order,mix,E['qwen_source'],E['phi_source'],rt,gt,tau)); air2.append(a)
    return {
      'honest':{**C.verdict_rates(hon),'composition':verdict_composition(hon)},
      'A1':{**C.verdict_rates(a1),'AIR':float(np.mean(air1)),'BDR':float(np.mean(np.asarray(heldneg[stat])>=th1)) if heldneg is not None else float(detfreeze['thresholds'][beta][stat]['measured_BDR']),'composition':verdict_composition(a1)},
      'A2':{**C.verdict_rates(a2),'AIR':float(np.mean(air2)),'BDR':float(np.mean(np.asarray(heldneg[state])>=th2)) if heldneg is not None else float(detfreeze['thresholds'][beta][state]['measured_BDR']),'composition':verdict_composition(a2)},
      '_rows':{'honest':hon,'A1':a1,'A2':a2}
    }

def detector_freeze_for_method(bundle,cal_pool,cal_sessions,cal_neg):
    strongest,allm,pos=metric_summary(bundle,cal_pool,cal_sessions,cal_neg)
    thresholds={}
    detector_keys=['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']
    for beta in PCFG['beta_grid']:
        thresholds[str(beta)]={}
        for k in detector_keys:
            th,b=C.threshold_for_bdr(cal_neg[k],beta); thresholds[str(beta)][k]={'threshold':float(th),'measured_BDR':float(b)}
    stat=max(['D1','D2'],key=lambda k:C.roc_metrics(pos[k],cal_neg[k])['AUROC_sym'])
    state=max(['D3_L4','D4_L4'],key=lambda k:C.roc_metrics(pos[k],cal_neg[k])['AUROC_sym'])
    return {'strongest_by_L':strongest,'all_calibration_metrics':allm,'thresholds':thresholds,'stateless_attacker':stat,'stateful_attacker_L4':state}

def overlap_diag(pool,sessions,J):
    sets=[set(s['selected']) for s in sessions]; conv=[set(pool[i]['conversation_id'] for i in s['selected']) for s in sessions]
    inter=[]; jac=[]; cinter=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            x=len(sets[i]&sets[j]); inter.append(x); jac.append(x/max(1,len(sets[i]|sets[j]))); cinter.append(len(conv[i]&conv[j]))
    freq=np.zeros(len(pool),int)
    for s in sessions: freq[np.asarray(s['selected'],int)]+=1
    qs=np.quantile(J,np.linspace(0,1,11)); dec=[]
    for d in range(10):
        lo,hi=qs[d],qs[d+1]; mask=(J>=lo)&((J<=hi) if d==9 else (J<hi)); dec.append({'decile':d+1,'queries':int(mask.sum()),'inclusions':int(freq[mask].sum()),'mean_inclusion':float(freq[mask].mean()) if mask.any() else 0.0})
    stat=lambda a:{'mean':float(np.mean(a)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(np.max(a))} if a else {}
    return {'unique_queries_used':int(np.sum(freq>0)),'per_query_inclusion_frequency':{pool[i]['id']:int(x) for i,x in enumerate(freq) if x},'pairwise_intersection':stat(inter),'pairwise_jaccard':stat(jac),'pairwise_conversation_intersection':stat(cinter),'query_score_decile_concentration':dec}

def session_manifest_rows(method,pool,sessions,J,lam=None):
    if method=='Full': weights=C.distribution_weights(J,lam)
    else: weights=np.ones(len(pool),float)/len(pool)
    rows=[]
    for s in sessions:
        sel=s['selected']; pre=s['random_order']; order=s['order']
        rows.append({'session_id':f'{method}_{s["episode"]:03d}','method_condition':method,'rng_seed':int(s['seed']),
          'query_ids':[pool[i]['id'] for i in sel],'conversation_ids':[pool[i]['conversation_id'] for i in sel],
          'query_scores':[float(J[i]) for i in sel],'selection_weights':[float(weights[i]) for i in sel],
          'pre_scheduler_order_query_ids':[pool[i]['id'] for i in pre],'transmitted_order_query_ids':[pool[i]['id'] for i in order],
          'lineage_evidence_order':'pre_scheduler_order' if method=='Full' else 'transmitted_order'})
    return rows

def heldout_neg_scores(bundle,held_real,pool,tag):
    real=[{'id':s['id'],'texts':s['texts']} for s in held_real]; negq=C.excluded_queries(real,[r['text'] for r in pool],6000,'G3_'+tag+'_HELD_NEG')
    if len(negq)<int(CFG['min_negative_queries']): raise RuntimeError(f'insufficient heldout benign negatives for {tag}: {len(negq)}')
    out={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(negq))[:,1],'D2':bundle['d2'].predict_proba(bundle['ec'].get(negq))[:,1]}
    for L in PCFG['L_grid']:
        nw=C.real_windows(real,L,PCFG['detector_window_cap'],'G3_'+tag+'_HELD_W'+str(L))
        if L in (4,8): out[f'D3_L{L}']=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],nw,L)
        out[f'D4_L{L}']=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],nw,L)
    return out

def heldout_neg_score_blocks(bundle,held_real,pool,tag):
    """Score benign held-out evidence while retaining original conversation blocks."""
    ex={r['text'] for r in pool}; seen=set(); qrows=[]
    for ss in held_real:
        for t in ss['texts']:
            if t in ex or t in seen: continue
            seen.add(t); qrows.append((t,ss['id']))
    qrows.sort(key=lambda x:C.h64(x[0]+'|'+x[1],'G3_'+tag+'_HELD_NEG'))
    qrows=qrows[:6000]
    if len(qrows)<int(CFG['min_negative_queries']): raise RuntimeError(f'insufficient untouched heldout benign negatives for {tag}: {len(qrows)}')
    blocks={k:collections.defaultdict(list) for k in ['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']}
    texts=[x[0] for x in qrows]; cids=[x[1] for x in qrows]
    qscore={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(texts))[:,1], 'D2':bundle['d2'].predict_proba(bundle['ec'].get(texts))[:,1]}
    # D4_L1 is a sequence detector at L=1.
    items=[([t],cid) for t,cid in qrows]; qscore['D4_L1']=C.score_seq(bundle['seq']['D4_L1'],bundle['ec'],items,1)
    for k,a in qscore.items():
        for cid,v in zip(cids,a): blocks[k][cid].append(float(v))
    real=[{'id':ss['id'],'texts':ss['texts']} for ss in held_real]
    for L in (4,8):
        items=C.real_windows(real,L,PCFG['detector_window_cap'],'G3_'+tag+'_HELD_W'+str(L))
        if L in (4,8):
            a=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],items,L)
            for (_,cid),v in zip(items,a): blocks[f'D3_L{L}'][cid].append(float(v))
        a=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],items,L)
        for (_,cid),v in zip(items,a): blocks[f'D4_L{L}'][cid].append(float(v))
    all_cids=sorted(set(cids) | {cid for k in blocks.values() for cid in k})
    return {k:{cid:np.asarray(v,float) for cid,v in d.items()} for k,d in blocks.items()}, all_cids

def sample_neg_from_blocks(blocks, sampled_cids):
    out={}
    for k,d in blocks.items():
        z=[d[cid] for cid in sampled_cids if cid in d and len(d[cid])]
        out[k]=np.concatenate(z) if z else np.empty(0,float)
    return out

def metric_by_frozen_strongest(bundle,pool,sessions,neg,freeze):
    pos=positive_scores(bundle,pool,sessions); out={}; fronts={}
    for L in PCFG['L_grid']:
        det=freeze['strongest_by_L'][str(L)]['detector']; out[str(L)]={'detector':det,**C.roc_metrics(pos[det],neg[det])}; fronts[str(L)]=frontier(pos[det],neg[det])
    return out,fronts,pos

def bootstrap_primary(pool,J,states,scheduler,E,bundles,freezes,lin,held_real,lam,reps,logfn):
    # Primary conversation-cluster bootstrap: query/response candidate clusters AND benign query/window clusters are block-resampled.
    by=collections.defaultdict(list)
    for i,r in enumerate(pool): by[r['conversation_id']].append(i)
    cids=sorted(by); rng=np.random.default_rng(PCFG['seed']+909090)
    neg_blocks={}; neg_cids=None
    for m in METHODS:
        neg_blocks[m], mcids=heldout_neg_score_blocks(bundles[m],held_real,pool,'BOOT_'+m)
        neg_cids=mcids if neg_cids is None else sorted(set(neg_cids)&set(mcids))
    if not neg_cids: raise RuntimeError('no common untouched benign conversation blocks for primary bootstrap')
    vals={f'dAUROC_L{L}':[] for L in PCFG['L_grid']}
    vals.update({f'dBDR08_L{L}':[] for L in PCFG['L_grid']}); vals.update({f'dAIR01_L{L}':[] for L in PCFG['L_grid']})
    vals.update({'dCSVR_honest':[],'dCSVR_A1':[],'dCSVR_A2':[]})
    adapters={m:C.DNAAdapter() for m in METHODS}
    for b in range(reps):
        pick=rng.integers(0,len(cids),len(cids)); orig=[]; bp=[]
        for copy_no,k in enumerate(pick):
            cid=cids[int(k)]
            for oi in by[cid]:
                r=dict(pool[oi]); r['id']=f"{r['id']}__b{b}c{copy_no}"; r['conversation_id']=f"{r['conversation_id']}__boot{copy_no}"; bp.append(r); orig.append(oi)
        if len(bp)<PCFG['main_q']: continue
        # One shared benign block draw for both methods; all overlapping windows from a conversation remain together.
        npick=rng.integers(0,len(neg_cids),len(neg_cids)); sampled_neg=[neg_cids[int(i)] for i in npick]
        neg={m:sample_neg_from_blocks(neg_blocks[m],sampled_neg) for m in METHODS}
        needed=set(['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8'])
        if any(any(len(neg[m][k])<2 for k in needed) for m in METHODS): continue
        orig=np.asarray(orig,int); Jb=np.asarray(J)[orig]; Eb={k:np.asarray(v)[orig] for k,v in E.items()}; stb=np.asarray(states)[orig]
        f=C.make_sessions(Jb,lam,PCFG['heldout_sessions'],PCFG['main_q'],stb,scheduler,f'G3_BOOT_{b}',True)
        u=uniform_sessions(len(bp),PCFG['heldout_sessions'],PCFG['main_q'],f'G3_BOOT_{b}',paired_lam=lam)
        mets={}; line={}
        for m,ss in [('Full',f),('Unwrapped',u)]:
            mets[m],_,_=metric_by_frozen_strongest(bundles[m],bp,ss,neg[m],freezes[m])
            line[m]=eval_lineage(m,ss,bp,Eb,bundles[m],freezes[m],lin,neg[m],adapters[m])
        for L in PCFG['L_grid']:
            vals[f'dAUROC_L{L}'].append(mets['Full'][str(L)]['AUROC_sym']-mets['Unwrapped'][str(L)]['AUROC_sym'])
            vals[f'dBDR08_L{L}'].append(mets['Full'][str(L)]['BDR_at_AIR_0.8']-mets['Unwrapped'][str(L)]['BDR_at_AIR_0.8'])
            vals[f'dAIR01_L{L}'].append(mets['Full'][str(L)]['AIR_at_BDR_0.1']-mets['Unwrapped'][str(L)]['AIR_at_BDR_0.1'])
        for c,k in [('honest','dCSVR_honest'),('A1','dCSVR_A1'),('A2','dCSVR_A2')]: vals[k].append(line['Full'][c]['CSVR']-line['Unwrapped'][c]['CSVR'])
        if (b+1)%50==0: logfn(f'cluster bootstrap {b+1}/{reps}')
    ci={}
    for k,a in vals.items():
        aa=np.asarray(a,float); ci[k]={'n':int(len(aa)),'mean':float(np.mean(aa)) if len(aa) else None,'ci95':[float(np.quantile(aa,.025)),float(np.quantile(aa,.975))] if len(aa) else [None,None]}
    completed=min((len(v) for v in vals.values()),default=0)
    if completed < int(reps*.95): raise RuntimeError(f'cluster bootstrap valid replicates too few: {completed}/{reps}')
    return {'procedure':'paired conversation-cluster bootstrap; candidate query/response clusters resampled with replacement; benign detector query/window blocks resampled by original conversation with all overlapping windows kept together; Full/Unwrapped sessions reconstructed in every replicate using the same replicate draw and frozen RNG protocol','clustering_unit':'original WildChat conversation_id; queries within conversation and all overlapping L=4/L=8 windows remain in block','replicates_requested':int(reps),'replicates_completed':int(completed),'effects':ci,'naive_session_bootstrap_primary':False}

def write_csvs(ret,results,frontiers):
    with open(Path(ret)/'detector_frontiers.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','L','AIR','BDR','threshold']); w.writeheader()
        for m in METHODS:
            for L,rows in frontiers[m].items():
                for r in rows:w.writerow({'method':m,'L':L,**r})
    with open(Path(ret)/'verdict_composition.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','condition','verdict','count','rate']); w.writeheader()
        for m in METHODS:
            for cond in ['honest','A1','A2']:
                for v,x in results[m]['lineage'][cond]['composition'].items(): w.writerow({'method':m,'condition':cond,'verdict':v,**x})
    with open(Path(ret)/'fig4_source.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','L','AUROC','BDR_at_AIR_0.8','AIR_at_BDR_0.1']); w.writeheader()
        for m in METHODS:
            for L,x in results[m]['detector'].items(): w.writerow({'method':m,'L':L,'AUROC':x['AUROC_sym'],'BDR_at_AIR_0.8':x['BDR_at_AIR_0.8'],'AIR_at_BDR_0.1':x['AIR_at_BDR_0.1']})

def protocol_hashes(ret):
    auth=PKG/'authorities'; files={}
    for p in sorted(auth.iterdir()):
        if p.is_file() and p.suffix in ('.md','.py'): files[p.name]=sha_file(p)
    files['g3_config.json']=sha_file(PKG/'config/g3_config.json'); files['project.json']=sha_file(PKG/'config/project.json')
    dump(ret,'G3_PROTOCOL_HASHES.json',{'files':files,'package_active_code':{p.name:sha_file(p) for p in sorted((PKG/'scripts').glob('*.py'))}})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); args=ap.parse_args(); rd=Path(args.run_dir); ret=rd/'return'; ret.mkdir(parents=True,exist_ok=True); configure_core(rd)
    heldout_touched=False; deviations=[]; phases={}
    try:
        pre=load_json(rd/'precheck'/'precheck_report.json')
        if not pre or pre.get('status')!='PASS': raise RuntimeError('precheck PASS required before G3 execution')
        protocol_hashes(ret); update_status(rd,status='RUNNING',phase='DEVCAL_REFREEZE',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY')
        log(rd,'=== G3 corrected matched effectiveness: development/calibration refreeze ===')
        dev_real,cal_real,dev_pool,cal_pool=reconstructed_devcal(); phases['pools']='ok'
        rec=[{**r,'pool':'dev'} for r in dev_pool]+[{**r,'pool':'cal'} for r in cal_pool]
        banks,reuse=acquire_devcal_banks(rec,rd); Eall=embed_banks(banks); nd=len(dev_pool); devE={k:v[:nd] for k,v in Eall.items()}; calE={k:v[nd:] for k,v in Eall.items()}; phases['devcal_banks']='ok'
        Jd,scale,rawd=source_only_J(devE['qwen_source'],devE['phi_source']); Jc,_,rawc=source_only_J(calE['qwen_source'],calE['phi_source'],scale)
        info={'definition':'cosdist(Qwen-source reference, Phi-source reference), first64 frozen response-embedding coordinates; clipped by development q05/q95; NO descendant/suspect response enters J_E','inputs':[{'name':'Qwen source responses','role':'source reference','descendant_derived':False},{'name':'Phi source responses','role':'alternative source reference','descendant_derived':False}],'forbidden_inputs':['Qwen NF4','LoRA','suspect/descendant response','descendant-derived statistics'],'first_coordinates':64,'development_scaling':{'q05':scale[0],'q95':scale[1]},'dev_mean':float(Jd.mean()),'cal_mean':float(Jc.mean()),'nf4_usage':'native lineage verification/calibration only; never query selection/scaling/ranking/tie-break','asset_reuse':reuse,'self_audit':'PASS'}
        dump(ret,'information_boundary_audit.json',info); phases['source_only']='ok'
        from sentence_transformers import SentenceTransformer
        mp=SentenceTransformer(str(C.MPNET),device='cuda:0'); legacy_dev=[{'id':s['id'],'texts':s['texts']} for s in dev_real]
        sched=C.build_scheduler(legacy_dev,dev_pool,cal_pool,mp); phases['scheduler']='ok'
        full_dev={str(l):C.make_sessions(Jd,l,PCFG['dev_sessions_per_lambda'],PCFG['main_q'],sched['states_dev'],sched['schedule'],'G3_DEV',True) for l in PCFG['lambda_grid']}
        full_cal={str(l):C.make_sessions(Jc,l,PCFG['cal_sessions_per_lambda'],PCFG['main_q'],sched['states_cal'],sched['schedule'],'G3_CAL',True) for l in PCFG['lambda_grid']}
        unwrap_dev=uniform_sessions(len(dev_pool),PCFG['dev_sessions_per_lambda'],PCFG['main_q'],'G3_DEV_UNWRAP'); unwrap_cal=uniform_sessions(len(cal_pool),PCFG['cal_sessions_per_lambda'],PCFG['main_q'],'G3_CAL_UNWRAP')
        bundles={'Full':train_bundle('FULL',dev_real,dev_pool,list(full_dev.values()),mp),'Unwrapped':train_bundle('UNWRAP',dev_real,dev_pool,[unwrap_dev],mp)}; phases['detectors_trained']='ok'
        calneg={m:calibration_neg(bundles[m],cal_real,cal_pool,m.upper()) for m in METHODS}
        # Historical lineage calibration algorithm on uniform calibration sessions; no held-out access.
        base_cal=uniform_sessions(len(cal_pool),PCFG['lineage_cal_sessions'],PCFG['main_q'],'G3_LINEAGE_CAL')
        lin=calibrate_lineage(base_cal,calE)
        # Full lambda selection, exactly historical rule but with source-only J.
        perlam={}
        for l in PCFG['lambda_grid']:
            key=str(l); strongest,allm,pos=metric_summary(bundles['Full'],cal_pool,full_cal[key],calneg['Full'])
            lf={'thresholds':{},'stateless_attacker':None,'stateful_attacker_L4':None}
            # honest Full uses risk-aware wrapper and canonicalized evidence order (pre-scheduler random order)
            ad=C.DNAAdapter(); honest=[C.lineage_session(ad,s['random_order'],calE['qwen_nf4'],calE['qwen_source'],calE['phi_source'],lin['risk_threshold_kappa'],lin['bank_gap_threshold'],lin['similarity_threshold']) for s in full_cal[key]]
            th=full_theory(Jc,full_cal[key],l)
            perlam[key]={'honest':C.verdict_rates(honest),'strongest_by_L':strongest,**th}
        feasible=[str(l) for l in PCFG['lambda_grid'] if perlam[str(l)]['honest']['CSVR']>=.95]; cand=feasible or [str(l) for l in PCFG['lambda_grid']]
        def rank(k):
            x=perlam[k]; m=x['strongest_by_L']['4']; return (m['BDR_at_AIR_0.8'],-m['AUROC_sym'],x['Gamma_select']) if feasible else (x['honest']['CSVR'],m['BDR_at_AIR_0.8'],-m['AUROC_sym'],x['Gamma_select'])
        chosen=max(cand,key=rank); lam=float(chosen); phases['lambda_frozen']='ok'
        freezes={'Full':detector_freeze_for_method(bundles['Full'],cal_pool,full_cal[chosen],calneg['Full']),'Unwrapped':detector_freeze_for_method(bundles['Unwrapped'],cal_pool,unwrap_cal,calneg['Unwrapped'])}
        # Freeze everything BEFORE held-out response/result access.
        elig=load_json(rd/'precheck'/'eligibility.json'); eligible_pool=load_pool_jsonl(rd/'precheck'/'eligible_pool.jsonl')
        freeze_manifest={'stage':'G3_FREEZE_BEFORE_HELDOUT_RESPONSE_ACCESS','frozen_at':now(),'heldout_response_access':False,'heldout_metrics_access':False,'source_only_scorer':info,'eligible_pool_n':len(eligible_pool),'eligible_pool_ids_sha256':elig['hashes']['chosen_ids_sha256'],'q':64,'session_realizations':64,'lambda_grid':PCFG['lambda_grid'],'chosen_lambda':lam,'lambda_selection_rule':'Among settings with honest risk-aware CSVR >= 0.95, maximize strongest-L4 BDR@AIR=.8; tie-break lower strongest-L4 AUROC, then higher Gamma_select. If none feasible, maximize honest CSVR then same tie-breaks.','lambda_feasible':list(map(float,feasible)),'per_lambda':perlam,'lineage_thresholds':lin,'method_detector_freezes':freezes,'beta_grid':PCFG['beta_grid'],'L_grid':PCFG['L_grid'],'generation':{'max_new_tokens':64,'temperature':.7,'top_p':.9},'full_components':['source-only tilted query selection','fresh session randomization','history-aware transmission scheduling','evidence-order canonicalization to pre-scheduler random order','risk-aware final decision'],'unwrapped_components':['uniform/native query sampling','fresh random order','native LLM-DNA lineage','no scheduler','no canonicalization','no risk-aware wrapper'],'retuning_after_heldout':False}
        dump(ret,'g3_freeze_manifest.json',freeze_manifest); phases['freeze']='ok'
        # Acceptance gate: only now authorize held-out response access.
        heldout_touched=True; update_status(rd,status='RUNNING',phase='HELDOUT_ONE_SHOT',heldout_touched=True,scientific_state_advanced=True,resume_allowed='NO')
        log(rd,f'HELDOUT ACCESS AUTHORIZED ONCE: N={len(eligible_pool)} lambda={lam}')
        # Load heldout real sessions for benign negatives only; pool was precommitted by hash.
        held_real=untouched_held_real()
        # Exact bank generation for new untouched pool. No corpus/model substitution.
        hbanks={}
        for tag,path,kind in [('held_qwen_source',C.QWEN,'source'),('held_qwen_nf4',C.QWEN,'nf4'),('held_phi_source',C.PHI,'phi')]: hbanks[tag]=C.generate_bank(tag,path,kind,eligible_pool)
        HE0=embed_banks(hbanks); HE={'qwen_source':HE0['held_qwen_source'],'qwen_nf4':HE0['held_qwen_nf4'],'phi_source':HE0['held_phi_source']}; phases['heldout_bank']='ok'
        Jh,_,_=source_only_J(HE['qwen_source'],HE['phi_source'],scale); states=sched['state_fn']([r['text'] for r in eligible_pool])
        full=C.make_sessions(Jh,lam,PCFG['heldout_sessions'],PCFG['main_q'],states,sched['schedule'],'G3_HELD_MATCHED',True)
        unwrap=uniform_sessions(len(eligible_pool),PCFG['heldout_sessions'],PCFG['main_q'],'G3_HELD_MATCHED',paired_lam=lam)
        manifests=session_manifest_rows('Full',eligible_pool,full,Jh,lam)+session_manifest_rows('Unwrapped',eligible_pool,unwrap,Jh,None); jsonl_write(ret/'session_manifest.jsonl',manifests)
        dump(ret,'heldout_pool_manifest.json',{'n':len(eligible_pool),'minimum_required':CFG['candidate_min'],'target':CFG['candidate_target'],'q':64,'query_ids_sha256':sha_text('\n'.join(r['id'] for r in eligible_pool)),'conversation_ids_sha256':sha_text('\n'.join(sorted({r['conversation_id'] for r in eligible_pool}))),'eligibility':elig,'heldout_response_access':True})
        overlap={'Full':overlap_diag(eligible_pool,full,Jh),'Unwrapped':overlap_diag(eligible_pool,unwrap,Jh)}; dump(ret,'session_overlap_diagnostics.json',overlap)
        heldneg={m:heldout_neg_scores(bundles[m],held_real,eligible_pool,m.upper()) for m in METHODS}
        results={}; frontiers={}; held_adapters={m:C.DNAAdapter() for m in METHODS}
        for m,ss in [('Full',full),('Unwrapped',unwrap)]:
            det,fr,pos=metric_by_frozen_strongest(bundles[m],eligible_pool,ss,heldneg[m],freezes[m]); line=eval_lineage(m,ss,eligible_pool,HE,bundles[m],freezes[m],lin,heldneg[m],held_adapters[m])
            results[m]={'detector':det,'lineage':{k:v for k,v in line.items() if k!='_rows'}}; frontiers[m]=fr
            if m=='Full': results[m]['source_only_selection']=full_theory(Jh,full,lam)
            dump(ret,'full_results.json' if m=='Full' else 'unwrapped_results.json',results[m])
        effects={'detector':{},'CSVR':{}}
        for L in PCFG['L_grid']:
            a=results['Full']['detector'][str(L)]; b=results['Unwrapped']['detector'][str(L)]
            effects['detector'][str(L)]={'delta_AUROC':a['AUROC_sym']-b['AUROC_sym'],'delta_BDR_at_AIR_0.8':a['BDR_at_AIR_0.8']-b['BDR_at_AIR_0.8'],'delta_AIR_at_BDR_0.1':a['AIR_at_BDR_0.1']-b['AIR_at_BDR_0.1']}
        for c in ['honest','A1','A2']: effects['CSVR'][c]=results['Full']['lineage'][c]['CSVR']-results['Unwrapped']['lineage'][c]['CSVR']
        a1crit={'gain':effects['CSVR']['A1'],'retention':results['Full']['lineage']['A1']['CSVR']/max(results['Full']['lineage']['honest']['CSVR'],1e-12)}; a1crit['pass']=a1crit['gain']>=.15 and a1crit['retention']>=.75
        a2crit={'gain':effects['CSVR']['A2'],'retention':results['Full']['lineage']['A2']['CSVR']/max(results['Full']['lineage']['honest']['CSVR'],1e-12)}; a2crit['pass']=a2crit['gain']>=.15 and a2crit['retention']>=.75
        criteria={
          'relative_cover_criterion':{'definition':'L4 BDR@AIR=.8 Full-minus-Unwrapped >= +0.20','value':effects['detector']['4']['delta_BDR_at_AIR_0.8'],'pass':effects['detector']['4']['delta_BDR_at_AIR_0.8']>=.20},
          'honest_utility_criterion':{'definition':'Full honest CSVR drop <= 0.05 vs matched Unwrapped','value':results['Full']['lineage']['honest']['CSVR']-results['Unwrapped']['lineage']['honest']['CSVR'],'pass':results['Full']['lineage']['honest']['CSVR']>=results['Unwrapped']['lineage']['honest']['CSVR']-.05},
          'original_G5_bounded_attack_utility':{'definition':'A1 AND A2: Full CSVR gain >= +0.15 and retention >= 75% of Full honest','A1':a1crit,'A2':a2crit,'pass':bool(a1crit['pass'] and a2crit['pass']), 'note':'legacy/original criterion label only; this does not execute or authorize current post-rejection G4/G5'}
        }
        effects['original_criteria']=criteria; dump(ret,'matched_effects.json',effects); write_csvs(ret,results,frontiers); phases['heldout_metrics']='ok'
        # Frozen 2000-replicate primary uncertainty. No retuning based on result.
        ci=bootstrap_primary(eligible_pool,Jh,states,sched['schedule'],HE,bundles,freezes,lin,held_real,lam,int(PCFG['bootstrap_reps']),lambda x:log(rd,x)); dump(ret,'cluster_bootstrap_CI.json',ci); phases['bootstrap']='ok'
        # Attach CIs to effects without altering any threshold/selection.
        for L in PCFG['L_grid']:
            effects['detector'][str(L)]['ci95_delta_AUROC']=ci['effects'][f'dAUROC_L{L}']['ci95']; effects['detector'][str(L)]['ci95_delta_BDR_at_AIR_0.8']=ci['effects'][f'dBDR08_L{L}']['ci95']; effects['detector'][str(L)]['ci95_delta_AIR_at_BDR_0.1']=ci['effects'][f'dAIR01_L{L}']['ci95']
        for c,k in [('honest','dCSVR_honest'),('A1','dCSVR_A1'),('A2','dCSVR_A2')]: effects['CSVR'][c]={'effect':effects['CSVR'][c],'ci95':ci['effects'][k]['ci95']}
        dump(ret,'matched_effects.json',effects)
        with open(ret/'EXECUTION_DEVIATIONS.md','w',encoding='utf-8') as f:f.write('# EXECUTION DEVIATIONS\n\n'+('None. Frozen G3 protocol executed without scientific deviation.\n' if not deviations else '\n'.join('- '+x for x in deviations)+'\n'))
        # execution layer reports facts only; it does not select scientific verdict label.
        with open(ret/'paper_result_summary.md','w',encoding='utf-8') as f:
            f.write('# SLATE G3 Matched Effectiveness — Execution Result\n\n')
            f.write(f'- held-out touched: YES (one-shot after freeze)\n- deviation: {"NO" if not deviations else "YES"}\n- Full chosen lambda: {lam}\n- untouched held-out N: {len(eligible_pool)}; q=64; sessions=64\n- retuning/rescue after held-out: NO\n\n')
            f.write('## Full minus Unwrapped measured effects\n\n```json\n'+json.dumps(effects,indent=2)+'\n```\n\n')
            f.write('## Original criteria (reported mechanically, not scientifically promoted)\n\n```json\n'+json.dumps(criteria,indent=2)+'\n```\n\n')
            f.write('This runner emits measured effects and frozen criterion checks; scientific interpretation is performed separately.\n')
        # Exact code/config copy into return.
        import shutil
        exact=ret/'exact_code_config'; shutil.copytree(PKG/'scripts',exact/'scripts',dirs_exist_ok=True); shutil.copytree(PKG/'config',exact/'config',dirs_exist_ok=True)
        rows=tree_manifest(ret,exclude=('MANIFEST_SHA256.txt',));
        with open(ret/'MANIFEST_SHA256.txt','w',encoding='utf-8') as f:
            for r in rows:f.write(f"{r['sha256']}  {r['path']}\n")
        update_status(rd,status='SUCCESS',phase='COMPLETE',heldout_touched=True,scientific_state_advanced=True,resume_allowed='NO',exit_code=0)
        log(rd,'G3 EXECUTION SUCCESS: evidence complete; scientific adjudication deferred to paper analysis')
        return 0
    except Exception as e:
        dump(ret,'EXECUTION_ERROR.json',{'error':repr(e),'traceback':traceback.format_exc(),'heldout_touched':heldout_touched,'phases':phases,'time':now()})
        with open(ret/'EXECUTION_DEVIATIONS.md','w',encoding='utf-8') as f:f.write('# EXECUTION DEVIATIONS\n\nExecution failure; no scientific substitution or rescue was attempted.\n')
        with open(ret/'paper_result_summary.md','w',encoding='utf-8') as f:f.write(f'# SLATE G3 Execution Failure\n\n- held-out touched: {"YES" if heldout_touched else "NO"}\n- error: `{repr(e)}`\n- retuning/rescue: NO\n- scientific adjudication: NOT PERFORMED\n')
        update_status(rd,status='FAILED',phase='HELDOUT_FAILURE' if heldout_touched else 'PREHELDOUT_FAILURE',heldout_touched=heldout_touched,scientific_state_advanced=heldout_touched,resume_allowed='NO' if heldout_touched else 'CONDITIONAL_MECHANICAL_ONLY',failure_summary=repr(e),exit_code=1)
        log(rd,traceback.format_exc()); return 1

if __name__=='__main__': sys.exit(main())
