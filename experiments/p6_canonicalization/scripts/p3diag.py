from __future__ import annotations
import json, os, sys, time, traceback, hashlib, collections, gc
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3c_llmdna_fullslate_heldout_confirmatory_v1'
RET=RUN/'return'; BANK=RUN/'banks'; EMB=RUN/'embeddings'
for p in [RUN,RET,BANK,EMB]: p.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
P3B_RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1'
P3B_RET=P3B_RUN/'return'
EMBEDDED=ROOT/'frozen_p3b'

sys.path.insert(0,str(ROOT/'scripts'))
import p3b_core as C
C.CFG=CFG
C.RUN=RUN; C.RET=RET; C.BANK=BANK; C.EMB=EMB; C.CACHE=RUN/'cache'; C.LOG=LOG
for p in [C.CACHE]: p.mkdir(parents=True,exist_ok=True)

EXPECTED={
 'p3c_freeze_manifest.json':'884e5ad9c049f0355bb977cac2fe2f44bd980aeaf8063096fe76d7ad2f256e1d',
 'candidate_pool_freeze.json':'3a14ee59d79274d1c24e35b6456cca1d8bd1542a2f18f3c8e798cfb89ae40389',
 'detector_freeze.json':'ca9591066a5f718491385a2df5351d3460ed9a7d16c97c84ccae310eab487c36',
 'evidence_model.json':'a13c763a7c20dc0e85bf68cdf005097ed2362448fd8d2ae210cc1c9a702d30c6',
 'lineage_thresholds.json':'94334ebdbb8bb7ce6c599f14945d671ef702c542e2f16e4ffb54f548d87d2381',
 'summary.json':'7e79063c3688ebdeb5fc6dcf78bfb9811944e35334702a81690644dce191f923'
}
FROZEN_STRONGEST={'1':'D4_L1','4':'D3_L4','8':'D3_L8'}

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(s+'\n')

def dump(name,obj):
    p=RET/name if isinstance(name,str) else Path(name)
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=str)

def load(p): return json.load(open(p,encoding='utf-8'))
def sem_hash_obj(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sem_hash_file(p): return sem_hash_obj(load(p))
def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()

def selftest():
    C.selftest()
    assert FROZEN_STRONGEST=={'1':'D4_L1','4':'D3_L4','8':'D3_L8'}
    for fn,h in EXPECTED.items():
        assert (EMBEDDED/fn).exists(), fn
        assert sem_hash_file(EMBEDDED/fn)==h, fn
    print('P3C_SELFTEST_OK')

def verify_freeze():
    st=load(P3B_RET/'status.json')
    if st.get('status')!='STAGEP3B_COMPLETE' or st.get('heldout_touched') is not False:
        raise RuntimeError('P3B is not a clean completed development freeze: '+json.dumps(st))
    audit={'p3b_status':st,'artifacts':{},'retuning_allowed':False}
    for fn,h in EXPECTED.items():
        sp=P3B_RET/fn; ep=EMBEDDED/fn
        if not sp.exists(): raise RuntimeError('missing P3B frozen artifact '+str(sp))
        hs=sem_hash_file(sp); he=sem_hash_file(ep)
        ok=(hs==he==h)
        audit['artifacts'][fn]={'server_semantic_sha256':hs,'embedded_semantic_sha256':he,'expected':h,'match':ok}
        if not ok: raise RuntimeError('P3B freeze hash mismatch for '+fn)
    fm=load(P3B_RET/'p3c_freeze_manifest.json')
    if fm.get('heldout_access_P3B') is not False or fm.get('lambda')!=0.05 or fm.get('q')!=64:
        raise RuntimeError('unexpected freeze manifest core settings')
    summ=load(P3B_RET/'summary.json')
    got={str(k):v['detector'] for k,v in summ['strongest_by_L'].items()}
    if got!=FROZEN_STRONGEST: raise RuntimeError('frozen strongest detector identities changed: '+json.dumps(got))
    audit['frozen_strongest']=got
    dump('freeze_audit.json',audit)
    return fm, load(P3B_RET/'detector_freeze.json'), load(P3B_RET/'evidence_model.json'), load(P3B_RET/'lineage_thresholds.json')

def load_p3b_embeddings(tag, records):
    p=P3B_RUN/'embeddings'/f'{tag}.npz'
    if not p.exists(): raise RuntimeError('missing P3B embedding cache '+str(p))
    z=np.load(p,allow_pickle=False); ids=list(z['ids'].astype(str)); E=np.asarray(z['E'],np.float32)
    mp={i:E[k] for k,i in enumerate(ids)}
    miss=[r['id'] for r in records if r['id'] not in mp]
    if miss: raise RuntimeError(f'P3B embedding cache missing {len(miss)} ids for {tag}')
    return np.stack([mp[r['id']] for r in records])

def devcal_pools():
    dev_real=C.sessions_for_split('development'); cal_real=C.sessions_for_split('calibration')
    dev_pool=C.freeze_pool(dev_real,CFG['candidate_dev'],'P3B_DEV_POOL')
    cal_pool=C.freeze_pool(cal_real,CFG['candidate_cal'],'P3B_CAL_POOL')
    fr=load(P3B_RET/'candidate_pool_freeze.json')
    dh=sha('\n'.join(x['id'] for x in dev_pool)); ch=sha('\n'.join(x['id'] for x in cal_pool))
    if dh!=fr['development_sha256'] or ch!=fr['calibration_sha256']:
        raise RuntimeError('development/calibration pool reconstruction hash mismatch')
    return dev_real,cal_real,dev_pool,cal_pool

def rebuild_frozen_detectors(fm,detfreeze,evidence):
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    dev_real,cal_real,dev_pool,cal_pool=devcal_pools()
    records=dev_pool+cal_pool; ndev=len(dev_pool)
    embs={k:load_p3b_embeddings(k,records) for k in ['qwen_source','qwen_nf4','phi_source']}
    devE={k:v[:ndev] for k,v in embs.items()}; calE={k:v[ndev:] for k,v in embs.items()}
    lo=float(fm['evidence_scaling']['q05']); hi=float(fm['evidence_scaling']['q95']); den=max(hi-lo,1e-8)
    raw=(1-C.cosrow(devE['qwen_source'][:,:64],devE['phi_source'][:,:64]))-(1-C.cosrow(devE['qwen_source'][:,:64],devE['qwen_nf4'][:,:64]))
    Jd=np.clip((raw-lo)/den,0,1)
    cr=(1-C.cosrow(calE['qwen_source'][:,:64],calE['phi_source'][:,:64]))-(1-C.cosrow(calE['qwen_source'][:,:64],calE['qwen_nf4'][:,:64]))
    Jc=np.clip((cr-lo)/den,0,1)
    if abs(float(Jd.mean())-float(evidence['dev_mean']))>1e-7 or abs(float(Jc.mean())-float(evidence['cal_mean']))>1e-7:
        raise RuntimeError('frozen evidence reconstruction mismatch')

    mp=SentenceTransformer(str(C.MPNET),device='cuda:0'); sched=C.build_scheduler(dev_real,dev_pool,cal_pool,mp); ec=C.ECache(mp)
    dev_sessions={str(l):C.make_sessions(Jd,l,CFG['dev_sessions_per_lambda'],CFG['main_q'],sched['states_dev'],sched['schedule'],'DEV',True) for l in CFG['lambda_grid']}
    cal_sessions={str(l):C.make_sessions(Jc,l,CFG['cal_sessions_per_lambda'],CFG['main_q'],sched['states_cal'],sched['schedule'],'CAL',True) for l in CFG['lambda_grid']}

    posq=[]
    for l in CFG['lambda_grid']: posq.extend(C.flatten_queries(dev_pool,dev_sessions[str(l)]))
    posq=posq[:CFG['detector_query_cap']]
    negq=C.excluded_queries(dev_real,[r['text'] for r in dev_pool],CFG['detector_query_cap'],'P3B_DNEG')
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=100000,sublinear_tf=True,strip_accents='unicode')
    X=vec.fit_transform(posq+negq); y=np.r_[np.ones(len(posq)),np.zeros(len(negq))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=CFG['seed']).fit(X,y)
    d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=CFG['seed']).fit(np.r_[ec.get(posq),ec.get(negq)],y)
    seq={}; pooled=[s for l in CFG['lambda_grid'] for s in dev_sessions[str(l)]]
    for L in CFG['L_grid']:
        pos=C.audit_items(dev_pool,pooled,L)[:CFG['detector_window_cap']]
        negw=C.real_windows(dev_real,L,CFG['detector_window_cap'],'P3B_NEGW'+str(L))
        if L in (4,8): seq[f'D3_L{L}']=C.fit_seq(ec,pos,negw,L,CFG['d3_hidden'],False,CFG['seed']+L)
        seq[f'D4_L{L}']=C.fit_seq(ec,pos,negw,L,CFG['d4_hidden'],True,CFG['seed']+100+L)

    cal_negq=C.excluded_queries(cal_real,[r['text'] for r in cal_pool],6000,'P3B_CALNEG')
    rawneg={'D1':d1.predict_proba(vec.transform(cal_negq))[:,1],'D2':d2.predict_proba(ec.get(cal_negq))[:,1]}
    for L in CFG['L_grid']:
        nw=C.real_windows(cal_real,L,CFG['detector_window_cap'],'P3B_CALW'+str(L))
        if L in (4,8): rawneg[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,nw,L)
        rawneg[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,nw,L)

    chosen=str(fm['lambda']); sess=cal_sessions[chosen]; qs=C.flatten_queries(cal_pool,sess)
    pos_scores={'D1':d1.predict_proba(vec.transform(qs))[:,1],'D2':d2.predict_proba(ec.get(qs))[:,1]}
    for L in CFG['L_grid']:
        ai=C.audit_items(cal_pool,sess,L)
        if L in (4,8): pos_scores[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,ai,L)
        pos_scores[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,ai,L)

    p3bs=load(P3B_RET/'summary.json')['strongest_by_L']; audit={'status':'PASS','strongest':{},'frozen_threshold_bdr':{}}
    for L,det in FROZEN_STRONGEST.items():
        met=C.roc_metrics(pos_scores[det],rawneg[det]); exp=float(p3bs[L]['AUROC_sym']); diff=abs(float(met['AUROC_sym'])-exp)
        audit['strongest'][L]={'detector':det,'rebuilt':met,'p3b_expected_AUROC_sym':exp,'abs_diff':diff}
        if diff>CFG['detector_rebuild_auc_tolerance']:
            audit['status']='FAIL'; dump('detector_rebuild_audit.json',audit); raise RuntimeError(f'detector rebuild AUROC audit failed L={L}: diff={diff}')
    for beta in CFG['beta_grid']:
        b=str(beta); audit['frozen_threshold_bdr'][b]={}
        for det in ['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']:
            th=float(fm['detector_thresholds'][b][det]['threshold']); actual=float(np.mean(rawneg[det]>=th)); exp=float(fm['detector_thresholds'][b][det]['measured_BDR']); diff=abs(actual-exp)
            audit['frozen_threshold_bdr'][b][det]={'actual':actual,'expected':exp,'abs_diff':diff}
            if diff>CFG['detector_rebuild_bdr_tolerance']:
                audit['status']='FAIL'; dump('detector_rebuild_audit.json',audit); raise RuntimeError(f'detector rebuild frozen-threshold BDR audit failed {b}/{det}: diff={diff}')
    dump('detector_rebuild_audit.json',audit)
    return {'dev_real':dev_real,'cal_real':cal_real,'dev_pool':dev_pool,'cal_pool':cal_pool,'Jd':Jd,'Jc':Jc,'mp':mp,'sched':sched,'ec':ec,'vec':vec,'d1':d1,'d2':d2,'seq':seq,'rawneg_cal':rawneg}

def heldout_sessions_all():
    by=collections.defaultdict(list)
    for r in C.load_jsonl(C.TRAFFIC):
        if r.get('split')!='heldout': continue
        t=str(r.get('text','')).strip()
        if C.valid_text(t): by[str(r['conversation_id'])].append((int(r.get('turn_index',0)),t))
    out=[]
    for cid,z in by.items():
        texts=[x[1] for x in sorted(z)]
        if texts: out.append({'id':cid,'texts':texts})
    out.sort(key=lambda x:C.h64(x['id'],'P3C_heldout'))
    return out

def bootstrap_roc(pos,neg,reps=1000):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float); rng=np.random.default_rng(CFG['seed']+991); vals=[]
    for _ in range(reps):
        p=pos[rng.integers(0,len(pos),len(pos))]; n=neg[rng.integers(0,len(neg),len(neg))]
        try: vals.append(C.roc_metrics(p,n))
        except Exception: continue
    if not vals: return {}
    out={}
    for k in ['AUROC_sym','BDR_at_AIR_0.8','AIR_at_BDR_0.1']:
        a=np.asarray([v[k] for v in vals],float); out[k]=[float(np.quantile(a,.025)),float(np.quantile(a,.975))]
    return out

def paired_delta_ci(a,b,reps=2000):
    a=np.asarray(a,float); b=np.asarray(b,float); assert len(a)==len(b); rng=np.random.default_rng(CFG['seed']+12345); z=[]
    for _ in range(reps):
        ii=rng.integers(0,len(a),len(a)); z.append(float(np.mean(a[ii]-b[ii])))
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def main():
    if '--selftest' in sys.argv: selftest(); return
    heldout_touched=False; phases={}; log('=== StageP3C LLM-DNA Full-SLATE Held-out Confirmatory ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        fm,detfreeze,evidence,lin=verify_freeze(); phases['freeze_audit']='ok'
        bundle=rebuild_frozen_detectors(fm,detfreeze,evidence); phases['detector_rebuild']='ok'
        # This is the only transition that authorizes held-out access.
        heldout_touched=True; dump('status.json',{'status':'RUNNING_HELDOUT','phases':phases,'heldout_touched':True})
        hreal=heldout_sessions_all()
        hpool=C.freeze_pool(hreal,CFG['candidate_heldout'],'P3C_HELDOUT_POOL')
        if len(hpool)<CFG['candidate_heldout']: raise RuntimeError('insufficient held-out candidate pool')
        devtexts={x['text'] for x in bundle['dev_pool']}; caltexts={x['text'] for x in bundle['cal_pool']}; htexts={x['text'] for x in hpool}
        overlap_dev=len(htexts & devtexts); overlap_cal=len(htexts & caltexts)
        if overlap_dev or overlap_cal: raise RuntimeError(f'held-out text overlap dev={overlap_dev}, cal={overlap_cal}')
        hp={'heldout_n':len(hpool),'q':CFG['main_q'],'heldout_sha256':sha('\n'.join(x['id'] for x in hpool)),'overlap_dev_text':overlap_dev,'overlap_cal_text':overlap_cal,'split':'heldout only','selection_salt':'P3C_HELDOUT_POOL'}; dump('heldout_pool.json',hp); phases['heldout_pool']='ok'

        banks={}
        for tag,path,kind in [('held_qwen_source',C.QWEN,'source'),('held_qwen_nf4',C.QWEN,'nf4'),('held_phi_source',C.PHI,'phi')]:
            banks[tag]=C.generate_bank(tag,path,kind,hpool)
        dump('heldout_response_bank_summary.json',{k:{'n':len(v),'sha256':sha('\n'.join(r['id']+'|'+r['response'] for r in v)),'mean_response_chars':float(np.mean([r['response_chars'] for r in v]))} for k,v in banks.items()}); phases['heldout_responses']='ok'

        from sentence_transformers import SentenceTransformer
        enc=SentenceTransformer(str(C.ENC),device='cuda:0')
        E={k:C.embed_bank(k,v,enc) for k,v in banks.items()}; del enc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
        src=E['held_qwen_source']; nf4=E['held_qwen_nf4']; phi=E['held_phi_source']
        lo=float(fm['evidence_scaling']['q05']); hi=float(fm['evidence_scaling']['q95']); den=max(hi-lo,1e-8)
        raw=(1-C.cosrow(src[:,:64],phi[:,:64]))-(1-C.cosrow(src[:,:64],nf4[:,:64])); Jh=np.clip((raw-lo)/den,0,1)
        hstates=bundle['sched']['state_fn']([r['text'] for r in hpool])
        hsess=C.make_sessions(Jh,float(fm['lambda']),CFG['heldout_sessions'],CFG['main_q'],hstates,bundle['sched']['schedule'],'HELDOUT_CONFIRM',True)
        hrand=C.make_sessions(Jh,float(fm['lambda']),CFG['heldout_sessions'],CFG['main_q'],hstates,bundle['sched']['schedule'],'HELDOUT_CONFIRM',False)

        negq=C.excluded_queries(hreal,[r['text'] for r in hpool],6000,'P3C_HELD_NEG')
        if len(negq)<100: raise RuntimeError('insufficient held-out benign negatives')
        d1,d2,vec,ec,seq=bundle['d1'],bundle['d2'],bundle['vec'],bundle['ec'],bundle['seq']
        rawneg={'D1':d1.predict_proba(vec.transform(negq))[:,1],'D2':d2.predict_proba(ec.get(negq))[:,1]}
        posq=C.flatten_queries(hpool,hsess); poss={'D1':d1.predict_proba(vec.transform(posq))[:,1],'D2':d2.predict_proba(ec.get(posq))[:,1]}
        for L in CFG['L_grid']:
            nw=C.real_windows(hreal,L,CFG['detector_window_cap'],'P3C_HELD_W'+str(L)); ai=C.audit_items(hpool,hsess,L)
            if len(nw)<50: raise RuntimeError(f'insufficient heldout benign windows L={L}')
            if L in (4,8):
                rawneg[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,nw,L); poss[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,ai,L)
            rawneg[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,nw,L); poss[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,ai,L)
        phases['heldout_detector_scores']='ok'

        strongest={}; roc_ci={}
        for L,det in FROZEN_STRONGEST.items():
            strongest[L]={'detector':det,**C.roc_metrics(poss[det],rawneg[det])}; roc_ci[L]=bootstrap_roc(poss[det],rawneg[det],1000)

        transfer={}
        for beta in CFG['beta_grid']:
            b=str(beta); transfer[b]={}
            for det in ['D1','D2','D4_L1','D3_L4','D4_L4','D3_L8','D4_L8']:
                th=float(fm['detector_thresholds'][b][det]['threshold'])
                transfer[b][det]={'threshold':th,'heldout_BDR':float(np.mean(rawneg[det]>=th)),'heldout_AIR':float(np.mean(poss[det]>=th)),'nominal_BDR':float(beta)}
        dump('threshold_transfer.json',transfer)

        tau=float(fm['lineage_similarity_threshold']); risk_tau=float(fm['risk_threshold_kappa']); gap_tau=float(fm['bank_gap_threshold'])
        ad=C.DNAAdapter()
        honestrows=[C.lineage_session(ad,s['order'],nf4,src,phi,risk_tau,gap_tau,tau) for s in hsess]
        hon=C.verdict_rates(honestrows)
        attacks={}; mainrows=None
        stat=fm['stateless_attacker']; state=fm['stateful_attacker_L4']
        for beta in CFG['beta_grid']:
            b=str(beta); a1=[];a2=[];air1=[];air2=[]
            th1=float(fm['detector_thresholds'][b][stat]['threshold']); th2=float(fm['detector_thresholds'][b][state]['threshold'])
            for s in hsess:
                mix,a,_=C.attack_a1(s,hpool,d1,d2,vec,ec,stat,th1,nf4,phi); a1.append(C.lineage_session(ad,s['order'],mix,src,phi,risk_tau,gap_tau,tau)); air1.append(a)
                mix,a,_=C.attack_a2(s,hpool,seq[state],ec,4,th2,nf4,phi); a2.append(C.lineage_session(ad,s['order'],mix,src,phi,risk_tau,gap_tau,tau)); air2.append(a)
            attacks[b]={'A1':{**C.verdict_rates(a1),'AIR':float(np.mean(air1)),'BDR':transfer[b][stat]['heldout_BDR']},'A2':{**C.verdict_rates(a2),'AIR':float(np.mean(air2)),'BDR':transfer[b][state]['heldout_BDR']}}
            if abs(float(beta)-float(fm['main_beta']))<1e-12: mainrows={'A1':a1,'A2':a2,'AIR1':air1,'AIR2':air2}
        phases['heldout_lineage_attacks']='ok'

        # Held-out theory bridge; no fitting or thresholding on held-out.
        w=C.distribution_weights(Jh,float(fm['lambda'])); K=float(CFG['main_q']*np.sum(w*np.log(np.maximum(w*len(w),1e-300))))
        meanj=float(np.mean([s['mean_J'] for s in hsess])); gamma=meanj-float(Jh.mean())
        realdist={L:C.real_state_span_dist(hreal,bundle['sched']['state_fn'],L,bundle['sched']['S']) for L in CFG['L_grid']}; deltas={}
        for L in CFG['L_grid']:
            dr=C.state_span_dist([s['random_order'] for s in hrand],hstates,L,bundle['sched']['S']); ds=C.state_span_dist([s['order'] for s in hsess],hstates,L,bundle['sched']['S'])
            deltas[str(L)]={'delta_random':C.tv(dr,realdist[L]),'delta_sched_L':C.tv(ds,realdist[L]),'improvement':C.tv(dr,realdist[L])-C.tv(ds,realdist[L])}
        eps={}
        bmain=str(fm['main_beta'])
        for L,det in FROZEN_STRONGEST.items():
            tr=transfer[bmain][det]; eps[L]={'detector':det,'AIR':tr['heldout_AIR'],'BDR':tr['heldout_BDR'],'epsilon_D_L':float(abs(tr['heldout_AIR']-tr['heldout_BDR']))}
        theory={'q':CFG['main_q'],'lambda':float(fm['lambda']),'mean_J':meanj,'heldout_pool_mean_J':float(Jh.mean()),'Khat_q':K,'Gamma_select':gamma,'delta_sched':deltas,'epsilon_D_L':eps,'selection_KL_bound_reference_q_over_lambda':float(CFG['main_q']/float(fm['lambda']))}; dump('theory_bridge_heldout.json',theory)

        hbool=np.asarray([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in honestrows],float)
        a1bool=np.asarray([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in mainrows['A1']],float)
        a2bool=np.asarray([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in mainrows['A2']],float)
        ci={'honest_CSVR':C.ci_binary(hbool),'A1_beta10_CSVR':C.ci_binary(a1bool),'A2_beta10_CSVR':C.ci_binary(a2bool),'A1_beta10_AIR':C.ci_binary(mainrows['AIR1']),'A2_beta10_AIR':C.ci_binary(mainrows['AIR2']),'paired_honest_minus_A1_CSVR':paired_delta_ci(hbool,a1bool),'paired_honest_minus_A2_CSVR':paired_delta_ci(hbool,a2bool),'detector_ROC':roc_ci}
        metrics={'honest':hon,'attacks':attacks,'strongest_by_L':strongest,'bootstrap_CI':ci,'heldout_n':len(hpool),'sessions':len(hsess)}; dump('heldout_metrics.json',metrics)

        p={'stage':'P3C','development_or_confirmatory':'held-out confirmatory','retuning_after_P3B':False,'engine':'LLM-DNA/RepTrace 1.0.1','family':'Qwen3-4B-Base','descendant_type':'NF4 runtime quantization','method':'Full SLATE','q':64,'lambda':float(fm['lambda']),'honest_csvr':hon['CSVR'],'A1_csvr':attacks[bmain]['A1']['CSVR'],'A2_csvr':attacks[bmain]['A2']['CSVR'],'fsvr':{'honest':hon['FSVR'],'A1':attacks[bmain]['A1']['FSVR'],'A2':attacks[bmain]['A2']['FSVR']},'channel_risk':{'honest':hon['Channel_Risk'],'A1':attacks[bmain]['A1']['Channel_Risk'],'A2':attacks[bmain]['A2']['Channel_Risk']},'insufficient':{'honest':hon['Insufficient'],'A1':attacks[bmain]['A1']['Insufficient'],'A2':attacks[bmain]['A2']['Insufficient']},'L1_auroc':strongest['1']['AUROC_sym'],'L4_auroc':strongest['4']['AUROC_sym'],'L8_auroc':strongest['8']['AUROC_sym'],'bdr_at_air_08':strongest['4']['BDR_at_AIR_0.8'],'air_at_bdr_01':strongest['4']['AIR_at_BDR_0.1'],'beta10_air':{'A1':attacks[bmain]['A1']['AIR'],'A2':attacks[bmain]['A2']['AIR']},'beta10_bdr':{'A1':attacks[bmain]['A1']['BDR'],'A2':attacks[bmain]['A2']['BDR']},'beta10_csvr':{'A1':attacks[bmain]['A1']['CSVR'],'A2':attacks[bmain]['A2']['CSVR']},'mean_J':theory['mean_J'],'Khat_q':theory['Khat_q'],'gamma_select':theory['Gamma_select'],'delta_sched':theory['delta_sched'],'epsilon_D_L':theory['epsilon_D_L'],'bootstrap_CI':ci,'threshold_transfer':transfer,'heldout_overlap':{'dev':overlap_dev,'cal':overlap_cal},'claims_supported':['one-shot held-out Full-SLATE confirmatory with P3B settings frozen'],'next_required':'P3D channel-integrity only after assessing this return; then P4 SRP development/held-out'}; dump('paper_result_summary.json',p)
        summary={'workflow':CFG['workflow'],'status':'STAGEP3C_COMPLETE','heldout_touched':True,'freeze_reused':'P3B v1.2.1','honest':hon,'beta10':{'A1':attacks[bmain]['A1'],'A2':attacks[bmain]['A2']},'strongest_by_L':strongest,'next':'Assess P3C gates without retuning. Then P3D channel integrity and P4 SRP.'}; dump('summary.json',summary)
        phases['paper_summary']='ok'; dump('status.json',{'status':'STAGEP3C_COMPLETE','phases':phases,'heldout_touched':True,'retuned':False}); log(json.dumps(summary,indent=2))
    except Exception as e:
        dump('status.json',{'status':'STAGEP3C_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'heldout_touched':heldout_touched,'retuned':False})
        dump('summary.json',{'workflow':CFG['workflow'],'status':'FAILED','error':repr(e),'heldout_touched':heldout_touched,'retuned':False})
        dump('paper_result_summary.json',{'stage':'P3C','development_or_confirmatory':'held-out confirmatory','status':'FAILED','paper_facing_claims':False,'error':repr(e),'heldout_touched':heldout_touched})
        raise

if __name__=='__main__': main()
