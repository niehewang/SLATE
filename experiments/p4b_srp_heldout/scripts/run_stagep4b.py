from __future__ import annotations
import collections, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep4b_srp_fullslate_heldout_confirmatory_v1_1_engineering_recovery'
PREV_RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep4b_srp_fullslate_heldout_confirmatory_v1'
PREV_RET=PREV_RUN/'return'
RET=RUN/'return'
for p in [RUN,RET]: p.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
SHARED=Path.home()/'SLATE_TDSC_SERVER_SHARED'
P4A_RUN=SHARED/'runs/slate_stagep4a_srp_fullslate_devcal_freeze_v1'
P4A_RET=P4A_RUN/'return'
P3B_RUN=SHARED/'runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1'
P3B_RET=P3B_RUN/'return'; P3B_EMB=P3B_RUN/'embeddings'
P3C_RUN=SHARED/'runs/slate_stagep3c_llmdna_fullslate_heldout_confirmatory_v1'
P3C_RET=P3C_RUN/'return'; P3C_EMB=P3C_RUN/'embeddings'
EMBEDDED=ROOT/'frozen_p4a'

sys.path.insert(0,str(ROOT/'scripts'))
import p4a_core as C
C.CFG=CFG; C.LOG=LOG; C.RET=RET

EXPECTED={
 'p4b_freeze_manifest.json':'31bb7e7a073414ffa62609845a1a811a837f539a4fe04d4c8975972730ab4211',
 'detector_freeze.json':'5994835ca3779051a5867f2a55543d26d55feae67c446004d18a9302f21a6531',
 'evidence_model.json':'6d13574627883cab0c2678e566467f50c11a745990025c07a213cce739500658',
 'lineage_thresholds.json':'68844768f501245c4077891ffa399ade1fe56f238b8d707bdd800449c197e7ca',
 'summary.json':'8667568dac79c1db8ebd81b3a0e0c73ca105762141878c4844bde495b2faac31',
 'development_metrics.json':'c7366c5a8e6e4c234af510fe59dc920a2703af049c441d7a22dbccee9205f46d'
}

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(s+'\n')

def dump(name,obj):
    p=RET/name if isinstance(name,str) else Path(name); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=str)

def load(p): return json.load(open(p,encoding='utf-8'))
def sem_hash_obj(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sem_hash_file(p): return sem_hash_obj(load(p))
def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()

def score_key(det, scores):
    """Map frozen display detector IDs to the persisted score-dictionary keys.
    P4A stores L=1 strongest detector labels as D1_L1/D2_L1, while the
    score dictionaries intentionally use D1/D2 for stateless detectors.
    Stateful keys (e.g. D4_L1, D3_L4) are already canonical.
    """
    if det in scores:
        return det
    if det in ('D1_L1','D2_L1'):
        base=det[:-3]
        if base in scores:
            return base
    raise KeyError(f'frozen detector {det!r} has no score key; available={sorted(scores)}')

def verify_prior_failure():
    sp=PREV_RET/'status.json'
    hp=PREV_RET/'heldout_pool.json'
    if not sp.exists() or not hp.exists():
        raise RuntimeError('missing prior P4B v1 failed-run provenance; refuse engineering recovery')
    st=load(sp); hpj=load(hp)
    if st.get('status')!='STAGEP4B_FAILED' or st.get('heldout_touched') is not True or st.get('retuned') is not False:
        raise RuntimeError('prior P4B status is not the expected clean post-heldout engineering failure')
    if "KeyError('D1_L1')" not in str(st.get('error','')):
        raise RuntimeError('prior P4B failure is not the registered D1_L1 key-mapping bug')
    required={'freeze_audit':'ok','detector_rebuild':'ok','heldout_pool':'ok','heldout_embedding_reuse':'ok','heldout_detector_scores':'ok'}
    ph=st.get('phases',{})
    if any(ph.get(k)!=v for k,v in required.items()):
        raise RuntimeError('prior P4B did not reach the registered post-score failure boundary')
    p3c=P3C_RET/'heldout_pool.json'
    if not p3c.exists(): raise RuntimeError('missing P3C heldout pool provenance')
    p3=load(p3c)
    if hpj.get('heldout_sha256')!=p3.get('heldout_sha256'):
        raise RuntimeError('prior P4B heldout pool differs from frozen P3C heldout pool')
    audit={'status':'PASS','prior_status':st,'prior_heldout_pool_sha256':hpj.get('heldout_sha256'),'recovery_scope':'detector display-name -> score-key normalization only','retuning_allowed':False,'new_model_generation':False,'second_confirmatory_claim':False}
    dump('engineering_recovery_audit.json',audit)
    return audit

def selftest():
    C.selftest()
    for fn,h in EXPECTED.items():
        assert (EMBEDDED/fn).exists(), fn
        assert sem_hash_file(EMBEDDED/fn)==h, fn
    fm=load(EMBEDDED/'p4b_freeze_manifest.json')
    assert fm['q']==64 and float(fm['lambda'])==1.0 and fm['heldout_access_P4A'] is False
    assert fm['full_detectors']['stateless']=='D1' and fm['full_detectors']['stateful_L4']=='D4_L4'
    assert score_key('D1_L1',{'D1':[0.1]})=='D1'
    assert score_key('D2_L1',{'D2':[0.1]})=='D2'
    assert score_key('D4_L1',{'D4_L1':[0.1]})=='D4_L1'
    print('P4B_V1_1_ENGINEERING_RECOVERY_SELFTEST_OK')

def verify_freeze():
    st=load(P4A_RET/'status.json')
    if st.get('status')!='STAGEP4A_COMPLETE' or st.get('heldout_touched') is not False or st.get('retuned') is not False:
        raise RuntimeError('P4A is not a clean completed development freeze: '+json.dumps(st))
    audit={'p4a_status':st,'retuning_allowed':False,'artifacts':{}}
    for fn,h in EXPECTED.items():
        sp=P4A_RET/fn; ep=EMBEDDED/fn
        if not sp.exists(): raise RuntimeError('missing P4A frozen artifact '+str(sp))
        hs=sem_hash_file(sp); he=sem_hash_file(ep); ok=(hs==he==h)
        audit['artifacts'][fn]={'server_semantic_sha256':hs,'embedded_semantic_sha256':he,'expected':h,'match':ok}
        if not ok: raise RuntimeError('P4A freeze hash mismatch for '+fn)
    fm=load(P4A_RET/'p4b_freeze_manifest.json')
    if fm.get('heldout_access_P4A') is not False or int(fm.get('q',-1))!=64 or abs(float(fm.get('lambda',-1))-1.0)>1e-12:
        raise RuntimeError('unexpected P4A freeze manifest core settings')
    dump('freeze_audit.json',audit)
    return fm,load(P4A_RET/'detector_freeze.json'),load(P4A_RET/'evidence_model.json'),load(P4A_RET/'lineage_thresholds.json')

def load_p3b_embeddings(tag, records):
    p=P3B_EMB/f'{tag}.npz'
    if not p.exists(): raise RuntimeError('missing P3B embedding bank '+str(p))
    z=np.load(p,allow_pickle=False); ids=list(z['ids'].astype(str)); E=np.asarray(z['E'],np.float32); mp={i:E[k] for k,i in enumerate(ids)}
    miss=[r['id'] for r in records if r['id'] not in mp]
    if miss: raise RuntimeError(f'P3B embedding bank missing {len(miss)} ids for {tag}')
    return np.stack([mp[r['id']] for r in records])

def rebuild_frozen(fm,detfreeze,evidence):
    from sentence_transformers import SentenceTransformer
    dev_real=C.sessions_for_split('development'); cal_real=C.sessions_for_split('calibration')
    dev_pool=C.freeze_pool(dev_real,CFG['candidate_dev'],'P3B_DEV_POOL'); cal_pool=C.freeze_pool(cal_real,CFG['candidate_cal'],'P3B_CAL_POOL')
    dh=sha('\n'.join(r['id'] for r in dev_pool)); ch=sha('\n'.join(r['id'] for r in cal_pool))
    if dh!=fm['candidate_pool_hashes']['dev'] or ch!=fm['candidate_pool_hashes']['cal']:
        raise RuntimeError('P4A candidate-pool reconstruction hash mismatch')
    records=dev_pool+cal_pool; ndev=len(dev_pool)
    embs={k:load_p3b_embeddings(k,records) for k in ['qwen_source','qwen_nf4','phi_source']}
    devE={k:v[:ndev] for k,v in embs.items()}; calE={k:v[ndev:] for k,v in embs.items()}
    lo=float(fm['evidence_scaling']['q05']); hi=float(fm['evidence_scaling']['q95']); den=max(hi-lo,1e-8)
    Jd=np.clip(((1-C.cosrow(devE['qwen_source'],devE['phi_source']))-lo)/den,0,1)
    Jc=np.clip(((1-C.cosrow(calE['qwen_source'],calE['phi_source']))-lo)/den,0,1)
    if abs(float(Jd.mean())-float(evidence['dev_mean']))>1e-7 or abs(float(Jc.mean())-float(evidence['cal_mean']))>1e-7:
        raise RuntimeError('P4A SRP evidence reconstruction mismatch')
    mp=SentenceTransformer(str(C.MPNET),device='cuda:0'); sched=C.build_scheduler(dev_real,dev_pool,cal_pool,mp); ec=C.ECache(mp); C.EC_GLOBAL=ec
    dev_sessions={str(l):C.make_sessions(Jd,l,CFG['dev_sessions_per_lambda'],CFG['main_q'],sched['states_dev'],sched['schedule'],'DEV',True) for l in CFG['lambda_grid']}
    cal_sessions={str(l):C.make_sessions(Jc,l,CFG['cal_sessions_per_lambda'],CFG['main_q'],sched['states_cal'],sched['schedule'],'CAL',True) for l in CFG['lambda_grid']}
    unwrapped_dev=C.fixed_sessions(Jd,CFG['dev_sessions_per_lambda'],CFG['main_q'],'DEV'); unwrapped_cal=C.fixed_sessions(Jc,CFG['cal_sessions_per_lambda'],CFG['main_q'],'CAL')
    full_dev=[s for l in CFG['lambda_grid'] for s in dev_sessions[str(l)]]
    full=C.train_suite('FULL',dev_pool,dev_real,full_dev,ec,0); unwrap=C.train_suite('UNWRAPPED',dev_pool,dev_real,unwrapped_dev,ec,1000)
    full_neg,full_thr_rebuilt=C.calibrate_suite('FULL',full,cal_real,cal_pool,ec); unwrap_neg,unwrap_thr_rebuilt=C.calibrate_suite('UNWRAPPED',unwrap,cal_real,cal_pool,ec)
    chosen=str(float(fm['lambda']))
    fmet,fstrong,fraw=C.suite_metrics(full,full_neg,cal_pool,cal_sessions[chosen],ec)
    umet,ustrong,uraw=C.suite_metrics(unwrap,unwrap_neg,cal_pool,unwrapped_cal,ec)
    audit={'status':'PASS','full':{},'unwrapped':{},'frozen_threshold_bdr':{'full':{},'unwrapped':{}}}
    for side,strong,fr in [('full',fstrong,fm['full_detectors']['strongest_by_L']),('unwrapped',ustrong,fm['unwrapped_detectors']['strongest_by_L'])]:
        for L in ['1','4','8']:
            expdet=fr[L]['detector']; gotdet=strong[L]['detector']; exp=float(fr[L]['AUROC_sym']); got=float(strong[L]['AUROC_sym']); diff=abs(got-exp)
            audit[side][L]={'expected_detector':expdet,'rebuilt_detector':gotdet,'expected_AUROC_sym':exp,'rebuilt_AUROC_sym':got,'abs_diff':diff}
            if gotdet!=expdet or diff>CFG['detector_rebuild_auc_tolerance']:
                audit['status']='FAIL'; dump('detector_rebuild_audit.json',audit); raise RuntimeError(f'{side} detector rebuild audit failed L={L}')
    for side,neg,fr in [('full',full_neg,fm['full_detectors']),('unwrapped',unwrap_neg,fm['unwrapped_detectors'])]:
        for beta in CFG['beta_grid']:
            b=str(beta); audit['frozen_threshold_bdr'][side][b]={}
            for det,dd in fr['thresholds'][b].items():
                th=float(dd['threshold']); actual=float(np.mean(np.asarray(neg[det])>=th)); exp=float(dd['measured_BDR']); diff=abs(actual-exp)
                audit['frozen_threshold_bdr'][side][b][det]={'actual':actual,'expected':exp,'abs_diff':diff}
                if diff>CFG['detector_rebuild_bdr_tolerance']:
                    audit['status']='FAIL'; dump('detector_rebuild_audit.json',audit); raise RuntimeError(f'{side} frozen-threshold BDR audit failed {b}/{det}')
    # Attacker identities are frozen; no held-out reselection.
    if fm['full_detectors']['stateless']!=detfreeze['full_slate']['stateless'] or fm['full_detectors']['stateful_L4']!=detfreeze['full_slate']['stateful_L4']:
        raise RuntimeError('full attacker identity mismatch')
    if fm['unwrapped_detectors']['stateless']!=detfreeze['unwrapped_srp']['stateless'] or fm['unwrapped_detectors']['stateful_L4']!=detfreeze['unwrapped_srp']['stateful_L4']:
        raise RuntimeError('unwrapped attacker identity mismatch')
    dump('detector_rebuild_audit.json',audit)
    return {'dev_real':dev_real,'cal_real':cal_real,'dev_pool':dev_pool,'cal_pool':cal_pool,'Jd':Jd,'Jc':Jc,'sched':sched,'ec':ec,'full':full,'unwrap':unwrap,'full_neg_cal':full_neg,'unwrap_neg_cal':unwrap_neg,'calE':calE}

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

def load_p3c_held_embedding(tag,records):
    p=P3C_EMB/f'{tag}.npz'
    if not p.exists(): raise RuntimeError('missing P3C held-out embedding cache '+str(p)+'; do not regenerate/tune, rerun P3C artifact recovery if needed')
    z=np.load(p,allow_pickle=False); ids=list(z['ids'].astype(str)); E=np.asarray(z['E'],np.float32); mp={i:E[k] for k,i in enumerate(ids)}
    miss=[r['id'] for r in records if r['id'] not in mp]
    if miss: raise RuntimeError(f'P3C held-out embedding bank missing {len(miss)} ids for {tag}')
    return np.stack([mp[r['id']] for r in records]),{'path':str(p),'shape':list(E.shape),'ids_n':len(ids)}

def score_suite_held(suite,ec,hreal,hpool,sessions,salt):
    negq=C.excluded_queries(hreal,[r['text'] for r in hpool],6000,'P4B_COMMON_HELD_NEG')
    if len(negq)<100: raise RuntimeError('insufficient held-out benign negatives')
    rawneg={'D1':suite['d1'].predict_proba(suite['vec'].transform(negq))[:,1],'D2':suite['d2'].predict_proba(ec.get(negq))[:,1]}
    qs=C.flatten_queries(hpool,sessions); pos={'D1':suite['d1'].predict_proba(suite['vec'].transform(qs))[:,1],'D2':suite['d2'].predict_proba(ec.get(qs))[:,1]}
    for L in CFG['L_grid']:
        nw=C.real_windows(hreal,L,CFG['detector_window_cap'],f'P4B_COMMON_HELD_W{L}'); ai=C.audit_items(hpool,sessions,L)
        if len(nw)<50: raise RuntimeError(f'insufficient heldout benign windows L={L}')
        if L in (4,8):
            rawneg[f'D3_L{L}']=C.score_seq(suite['seq'][f'D3_L{L}'],ec,nw,L); pos[f'D3_L{L}']=C.score_seq(suite['seq'][f'D3_L{L}'],ec,ai,L)
        rawneg[f'D4_L{L}']=C.score_seq(suite['seq'][f'D4_L{L}'],ec,nw,L); pos[f'D4_L{L}']=C.score_seq(suite['seq'][f'D4_L{L}'],ec,ai,L)
    return rawneg,pos

def bootstrap_roc(pos,neg,reps=1000,seedoff=0):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float); rng=np.random.default_rng(CFG['seed']+991+seedoff); vals=[]
    for _ in range(reps):
        p=pos[rng.integers(0,len(pos),len(pos))]; n=neg[rng.integers(0,len(neg),len(neg))]
        try: vals.append(C.roc_metrics(p,n))
        except Exception: pass
    out={}
    for k in ['AUROC_sym','BDR_at_AIR_0.8','AIR_at_BDR_0.1']:
        a=np.asarray([v[k] for v in vals],float); out[k]=[float(np.quantile(a,.025)),float(np.quantile(a,.975))] if len(a) else [None,None]
    return out

def paired_delta_ci(a,b,reps=2000,seedoff=0):
    a=np.asarray(a,float); b=np.asarray(b,float); n=min(len(a),len(b)); a=a[:n]; b=b[:n]
    if n==0:return [None,None]
    rng=np.random.default_rng(CFG['seed']+12345+seedoff); z=[]
    for _ in range(reps):
        ii=rng.integers(0,n,n); z.append(float(np.mean(a[ii]-b[ii])))
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def attack_eval(sessions,hpool,suite,ec,rawneg,frozen,HE,tau_q,tau_p,risk_tau,gap_tau,use_risk):
    out={}; main=None; stat=frozen['stateless']; state=frozen['stateful_L4']
    for beta in CFG['beta_grid']:
        b=str(beta); th1=float(frozen['thresholds'][b][stat]['threshold']); th2=float(frozen['thresholds'][b][state]['threshold']); a1=[];a2=[];air1=[];air2=[]
        for s in sessions:
            mix,a,_=C.attack_a1(s,hpool,suite['d1'],suite['d2'],suite['vec'],ec,stat,th1,HE['qwen_nf4'],HE['phi_source']); a1.append(C.srp_session(s['order'],mix,HE['qwen_source'],HE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,use_risk)); air1.append(a)
            mix,a,_=C.attack_a2(s,hpool,suite['seq'][state],ec,4,th2,HE['qwen_nf4'],HE['phi_source']); a2.append(C.srp_session(s['order'],mix,HE['qwen_source'],HE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,use_risk)); air2.append(a)
        bdr1=float(np.mean(np.asarray(rawneg[stat])>=th1)); bdr2=float(np.mean(np.asarray(rawneg[state])>=th2))
        out[b]={'A1':{**C.verdict_rates(a1),'AIR':float(np.mean(air1)),'BDR':bdr1,'AIR_minus_BDR':float(np.mean(air1))-bdr1},'A2':{**C.verdict_rates(a2),'AIR':float(np.mean(air2)),'BDR':bdr2,'AIR_minus_BDR':float(np.mean(air2))-bdr2}}
        if abs(beta-CFG['main_beta'])<1e-12: main={'A1':a1,'A2':a2,'AIR1':air1,'AIR2':air2}
    honest=[C.srp_session(s['order'],HE['qwen_nf4'],HE['qwen_source'],HE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,use_risk) for s in sessions]
    return C.verdict_rates(honest),out,honest,main

def gate_decisions(fullstrong,unwrapstrong,fullhon,unwraphon,fullatt,unwrapatt):
    absbranch=(fullstrong['1']['AUROC_sym']<=.65 and fullstrong['4']['AUROC_sym']<=.70 and fullstrong['8']['AUROC_sym']<=.70)
    improvements={L:unwrapstrong[L]['AUROC_sym']-fullstrong[L]['AUROC_sym'] for L in ['1','4','8']}
    relbranch=all(v>=.15 for v in improvements.values())
    g2=bool(absbranch or relbranch)
    g3gain=float(fullstrong['4']['BDR_at_AIR_0.8']-unwrapstrong['4']['BDR_at_AIR_0.8']); g3=bool(g3gain>=.20)
    drop=float(unwraphon['CSVR']-fullhon['CSVR']); retain90=(fullhon['CSVR']>=.9*unwraphon['CSVR']) if unwraphon['CSVR']<.5 else False; g4=bool(drop<=.05 or retain90)
    b=str(CFG['main_beta']); g5parts={}
    for a in ['A1','A2']:
        gain=float(fullatt[b][a]['CSVR']-unwrapatt[b][a]['CSVR']); retention=(float(fullatt[b][a]['CSVR']/fullhon['CSVR']) if fullhon['CSVR']>0 else None); ok=(gain>=.15 and retention is not None and retention>=.75); g5parts[a]={'gain':gain,'retention_of_full_honest':retention,'pass':bool(ok)}
    g5=all(x['pass'] for x in g5parts.values())
    return {'G2_cover':{'pass':g2,'absolute_branch':absbranch,'relative_branch':relbranch,'improvements':improvements},'G3_security_cost':{'pass':g3,'L4_gain':g3gain},'G4_honest_utility':{'pass':g4,'full_CSVR':fullhon['CSVR'],'unwrapped_CSVR':unwraphon['CSVR'],'drop':drop,'unwrapped_zero_baseline':bool(unwraphon['CSVR']==0),'interpretation_note':'A mechanically passing G4 with zero unwrapped honest CSVR does not establish a useful native verifier.' if unwraphon['CSVR']==0 else ''},'G5_bounded_attack':{'pass':g5,'A1':g5parts['A1'],'A2':g5parts['A2']},'A0_baseline_note':{'unwrapped_honest_nonzero':bool(unwraphon['CSVR']>0),'note':'If unwrapped honest CSVR is zero, end-to-end protection gain claims are not eligible even if a relative formula is mechanically satisfied.'}}

def main():
    if '--selftest' in sys.argv: selftest(); return
    heldout_touched=False; phases={}; log('=== StageP4B SRP Full-SLATE Held-out Confirmatory ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        verify_prior_failure(); phases['engineering_recovery_audit']='ok'
        fm,detfreeze,evidence,lin=verify_freeze(); phases['freeze_audit']='ok'
        bundle=rebuild_frozen(fm,detfreeze,evidence); phases['detector_rebuild']='ok'
        # Only after all development/calibration audits pass may held-out be read.
        heldout_touched=True; dump('status.json',{'status':'RUNNING_HELDOUT','phases':phases,'heldout_touched':True,'retuned':False})
        hreal=heldout_sessions_all(); hpool=C.freeze_pool(hreal,CFG['candidate_heldout'],'P3C_HELDOUT_POOL')
        if len(hpool)!=CFG['candidate_heldout']: raise RuntimeError('insufficient held-out candidate pool')
        devtexts={r['text'] for r in bundle['dev_pool']}; caltexts={r['text'] for r in bundle['cal_pool']}; htexts={r['text'] for r in hpool}; od=len(htexts&devtexts); oc=len(htexts&caltexts)
        if od or oc: raise RuntimeError(f'held-out text overlap dev={od}, cal={oc}')
        hh=sha('\n'.join(r['id'] for r in hpool)); p3c_hp=P3C_RET/'heldout_pool.json'
        if p3c_hp.exists():
            old=load(p3c_hp)
            if old.get('heldout_sha256')!=hh: raise RuntimeError('P3C heldout pool hash mismatch; refuse to create a new confirmatory pool')
        dump('heldout_pool.json',{'heldout_n':len(hpool),'q':CFG['main_q'],'heldout_sha256':hh,'overlap_dev_text':od,'overlap_cal_text':oc,'split':'heldout only','reused_from':'P3C_HELDOUT_POOL','new_model_generation':False}); phases['heldout_pool']='ok'
        HE={}; reuse={}
        for tag in ['held_qwen_source','held_qwen_nf4','held_phi_source']:
            E,a=load_p3c_held_embedding(tag,hpool); HE[tag.replace('held_','')]=E; reuse[tag]=a
        dump('heldout_embedding_reuse_audit.json',{'source':'P3C heldout embeddings','heldout_pool_sha256':hh,'banks':reuse,'new_model_generation':False}); phases['heldout_embedding_reuse']='ok'
        lo=float(fm['evidence_scaling']['q05']); hi=float(fm['evidence_scaling']['q95']); den=max(hi-lo,1e-8); Jh=np.clip(((1-C.cosrow(HE['qwen_source'],HE['phi_source']))-lo)/den,0,1)
        hstates=bundle['sched']['state_fn']([r['text'] for r in hpool]); fullsess=C.make_sessions(Jh,float(fm['lambda']),CFG['heldout_sessions'],CFG['main_q'],hstates,bundle['sched']['schedule'],'P4B_HELDOUT_FULL',True); unwrapsess=C.fixed_sessions(Jh,CFG['heldout_sessions'],CFG['main_q'],'P4B_HELDOUT_UNWRAPPED')
        ec=bundle['ec']; C.EC_GLOBAL=ec
        fneg,fpos=score_suite_held(bundle['full'],ec,hreal,hpool,fullsess,'FULL'); uneg,upos=score_suite_held(bundle['unwrap'],ec,hreal,hpool,unwrapsess,'UNWRAPPED'); phases['heldout_detector_scores']='ok'
        fullstrong={}; unwrapstrong={}; rocci={'full':{},'unwrapped':{}}
        for side,pos,neg,fr,out,off in [('full',fpos,fneg,fm['full_detectors'],fullstrong,0),('unwrapped',upos,uneg,fm['unwrapped_detectors'],unwrapstrong,100)]:
            for L in ['1','4','8']:
                det=fr['strongest_by_L'][L]['detector']; sk=score_key(det,pos); nk=score_key(det,neg); out[L]={'detector':det,'score_key':sk,**C.roc_metrics(pos[sk],neg[nk])}; rocci[side][L]=bootstrap_roc(pos[sk],neg[nk],1000,off+int(L))
        transfer={'full':{},'unwrapped':{}}
        for side,pos,neg,fr in [('full',fpos,fneg,fm['full_detectors']),('unwrapped',upos,uneg,fm['unwrapped_detectors'])]:
            for beta in CFG['beta_grid']:
                b=str(beta); transfer[side][b]={}
                for det,dd in fr['thresholds'][b].items():
                    th=float(dd['threshold']); transfer[side][b][det]={'threshold':th,'heldout_BDR':float(np.mean(np.asarray(neg[det])>=th)),'heldout_AIR':float(np.mean(np.asarray(pos[det])>=th)),'nominal_BDR':float(beta)}
        dump('threshold_transfer.json',transfer)
        tau_q=float(fm['tau_qwen']); tau_p=float(fm['tau_phi']); risk_tau=float(fm['risk_threshold_kappa']); gap_tau=float(fm['bank_gap_threshold'])
        fullhon,fullatt,fullrows,fullmain=attack_eval(fullsess,hpool,bundle['full'],ec,fneg,fm['full_detectors'],HE,tau_q,tau_p,risk_tau,gap_tau,True)
        unwraphon,unwrapatt,unwraprows,unwrapmain=attack_eval(unwrapsess,hpool,bundle['unwrap'],ec,uneg,fm['unwrapped_detectors'],HE,tau_q,tau_p,risk_tau,gap_tau,False); phases['heldout_attacks']='ok'
        gates=gate_decisions(fullstrong,unwrapstrong,fullhon,unwraphon,fullatt,unwrapatt); dump('gate_decisions.json',gates)
        # Theory bridge for Full-SLATE only; parameters remain frozen.
        w=C.distribution_weights(Jh,float(fm['lambda'])); K=float(CFG['main_q']*np.sum(w*np.log(np.maximum(w*len(w),1e-300)))); meanj=float(np.mean([s['mean_J'] for s in fullsess])); gamma=meanj-float(Jh.mean()); realdist={L:C.real_state_span_dist(hreal,bundle['sched']['state_fn'],L,bundle['sched']['S']) for L in CFG['L_grid']}; deltas={}
        randomsess=C.make_sessions(Jh,float(fm['lambda']),CFG['heldout_sessions'],CFG['main_q'],hstates,bundle['sched']['schedule'],'P4B_HELDOUT_FULL',False)
        for L in CFG['L_grid']:
            dr=C.state_span_dist([s['random_order'] for s in randomsess],hstates,L,bundle['sched']['S']); ds=C.state_span_dist([s['order'] for s in fullsess],hstates,L,bundle['sched']['S']); deltas[str(L)]={'delta_random':C.tv(dr,realdist[L]),'delta_sched_L':C.tv(ds,realdist[L]),'improvement':C.tv(dr,realdist[L])-C.tv(ds,realdist[L])}
        eps={}; bmain=str(CFG['main_beta'])
        for L in ['1','4','8']:
            det=fm['full_detectors']['strongest_by_L'][L]['detector']; tk=score_key(det,transfer['full'][bmain]); tr=transfer['full'][bmain][tk]; eps[L]={'detector':det,'score_key':tk,'AIR':tr['heldout_AIR'],'BDR':tr['heldout_BDR'],'epsilon_D_L':float(abs(tr['heldout_AIR']-tr['heldout_BDR']))}
        theory={'q':CFG['main_q'],'lambda':float(fm['lambda']),'mean_J':meanj,'heldout_pool_mean_J':float(Jh.mean()),'Khat_q':K,'Gamma_select':gamma,'delta_sched':deltas,'epsilon_D_L':eps,'selection_KL_bound_reference_q_over_lambda':float(CFG['main_q']/float(fm['lambda']))}; dump('theory_bridge_heldout.json',theory)
        def vb(rows): return np.asarray([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in rows],float)
        fh,uh,fa1,fa2,ua1,ua2=vb(fullrows),vb(unwraprows),vb(fullmain['A1']),vb(fullmain['A2']),vb(unwrapmain['A1']),vb(unwrapmain['A2'])
        ci={'full_honest_CSVR':C.ci_binary(fh),'unwrapped_honest_CSVR':C.ci_binary(uh),'full_A1_beta10_CSVR':C.ci_binary(fa1),'full_A2_beta10_CSVR':C.ci_binary(fa2),'unwrapped_A1_beta10_CSVR':C.ci_binary(ua1),'unwrapped_A2_beta10_CSVR':C.ci_binary(ua2),'delta_full_minus_unwrapped_honest':paired_delta_ci(fh,uh,seedoff=1),'delta_full_minus_unwrapped_A1':paired_delta_ci(fa1,ua1,seedoff=2),'delta_full_minus_unwrapped_A2':paired_delta_ci(fa2,ua2,seedoff=3),'detector_ROC':rocci}
        metrics={'full_slate':{'honest':fullhon,'attacks':fullatt,'strongest_by_L':fullstrong},'unwrapped_srp':{'honest':unwraphon,'attacks':unwrapatt,'strongest_by_L':unwrapstrong},'bootstrap_CI':ci,'gates':gates,'heldout_n':len(hpool),'sessions':CFG['heldout_sessions']}; dump('heldout_metrics.json',metrics)
        paper={'stage':'P4B','development_or_confirmatory':'held-out confirmatory','retuning_after_P4A':False,'engine':'Semantic Response Profile (SRP)','family':'Qwen3-4B-Base','descendant_type':'NF4 runtime quantization','comparison':'Full SLATE vs matched unwrapped SRP','q':64,'lambda':float(fm['lambda']),'full_honest_csvr':fullhon['CSVR'],'unwrapped_honest_csvr':unwraphon['CSVR'],'full_beta10_csvr':{'A1':fullatt[bmain]['A1']['CSVR'],'A2':fullatt[bmain]['A2']['CSVR']},'unwrapped_beta10_csvr':{'A1':unwrapatt[bmain]['A1']['CSVR'],'A2':unwrapatt[bmain]['A2']['CSVR']},'full_auroc':{L:fullstrong[L]['AUROC_sym'] for L in ['1','4','8']},'unwrapped_auroc':{L:unwrapstrong[L]['AUROC_sym'] for L in ['1','4','8']},'full_bdr_at_air_08_L4':fullstrong['4']['BDR_at_AIR_0.8'],'unwrapped_bdr_at_air_08_L4':unwrapstrong['4']['BDR_at_AIR_0.8'],'gate_decisions':gates,'theory_bridge':theory,'bootstrap_CI':ci,'threshold_transfer':transfer,'heldout_overlap':{'dev':od,'cal':oc},'heldout_pool_sha256':hh,'reused_P3C_heldout_embeddings':True,'new_model_generation':False,'claim_scope':'one-shot held-out; no retuning; negative gates must be retained','next_required':'Assess P4B jointly with P3C. Do not retune. Then decide whether method claims need narrowing before minimal ablation/theory completion.'}; dump('paper_result_summary.json',paper)
        summary={'workflow':CFG['workflow'],'status':'STAGEP4B_COMPLETE','heldout_touched':True,'retuned':False,'new_model_generation':False,'reused_P3C_heldout_embeddings':True,'full_honest':fullhon,'unwrapped_honest':unwraphon,'beta10':{'full':{'A1':fullatt[bmain]['A1'],'A2':fullatt[bmain]['A2']},'unwrapped':{'A1':unwrapatt[bmain]['A1'],'A2':unwrapatt[bmain]['A2']}},'strongest_by_L':{'full':fullstrong,'unwrapped':unwrapstrong},'gates':gates,'next':'Joint P3/P4 gate assessment; no retuning.'}; dump('summary.json',summary); phases['paper_summary']='ok'; dump('status.json',{'status':'STAGEP4B_COMPLETE','phases':phases,'heldout_touched':True,'retuned':False,'new_model_generation':False}); log(json.dumps(summary,indent=2))
    except Exception as e:
        dump('status.json',{'status':'STAGEP4B_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'heldout_touched':heldout_touched,'retuned':False}); dump('summary.json',{'workflow':CFG['workflow'],'status':'FAILED','error':repr(e),'heldout_touched':heldout_touched,'retuned':False}); dump('paper_result_summary.json',{'stage':'P4B','development_or_confirmatory':'held-out confirmatory','status':'FAILED','paper_facing_claims':False,'error':repr(e),'heldout_touched':heldout_touched}); raise

if __name__=='__main__': main()
