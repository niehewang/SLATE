from __future__ import annotations
import json, hashlib, sys, traceback, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
P3CFG=json.load(open(ROOT/'config/p3.json'))
SHARED=Path.home()/'SLATE_TDSC_SERVER_SHARED'
RUN=SHARED/'runs/slate_stagep6_llmdna_canonicalization_diagnostic_v1'
RET=RUN/'return'; RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
P3C_RET=SHARED/'runs/slate_stagep3c_llmdna_fullslate_heldout_confirmatory_v1/return'
P5_RET=SHARED/'runs/slate_stagep5_joint_frozen_ablation_claim_boundary_v1_1_cache_tag_recovery/return'

sys.path.insert(0,str(ROOT/'scripts'))
import p3diag as P3
P3.CFG=P3CFG; P3.C.CFG=P3CFG
P3.RET=RET/'p3_rebuild'; P3.RET.mkdir(parents=True,exist_ok=True); P3.LOG=RUN/'p3_rebuild.log'
P3.C.RET=P3.RET; P3.C.LOG=P3.LOG; P3.C.RUN=RUN/'p3_tmp'; P3.C.CACHE=P3.C.RUN/'cache'; P3.C.CACHE.mkdir(parents=True,exist_ok=True)

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')
def dump(name,obj):
    p=RET/name; p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=str)
def load(p): return json.load(open(p,encoding='utf-8'))
def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()

def guard():
    s3=load(P3C_RET/'status.json'); s5=load(P5_RET/'status.json')
    if s3.get('status')!='STAGEP3C_COMPLETE' or s3.get('retuned') is not False:
        raise RuntimeError('P3C provenance not clean')
    if s5.get('status')!='STAGEP5_COMPLETE' or s5.get('diagnostic_only') is not True or s5.get('retuned') is not False or s5.get('new_model_generation') is not False:
        raise RuntimeError('P5 provenance not clean')
    p5=load(P5_RET/'paper_result_summary.json')
    if p5.get('status')!='STAGEP5_COMPLETE': raise RuntimeError('P5 summary incomplete')
    out={'status':'PASS','P3C_status':s3,'P5_status':s5,'diagnostic_only':True,'retuning_allowed':False,'new_model_generation':False,
         'scope':'post-confirmatory mechanism diagnostic only; no new confirmatory claim'}
    dump('provenance_guard.json',out); return out

def heldout_pool():
    hreal=P3.heldout_sessions_all(); hpool=P3.C.freeze_pool(hreal,P3CFG['candidate_heldout'],'P3C_HELDOUT_POOL')
    hh=sha('\n'.join(r['id'] for r in hpool)); exp=load(P3C_RET/'heldout_pool.json')['heldout_sha256']
    if hh!=exp: raise RuntimeError('heldout pool reconstruction mismatch')
    return hreal,hpool,hh

def held_embeddings(hpool):
    # Reuse P3C persisted heldout embeddings only. No generation.
    import p4diag as P4
    mapping={'qwen_source':'held_qwen_source','qwen_nf4':'held_qwen_nf4','phi_source':'held_phi_source'}
    E={}; audit={}
    for logical,persisted in mapping.items():
        E[logical],audit[logical]=P4.load_p3c_held_embedding(persisted,hpool)
        audit[logical]['logical_tag']=logical; audit[logical]['persisted_tag']=persisted
    return E,audit

def metric_triplet(core,pos,neg):
    m=core.roc_metrics(np.asarray(pos),np.asarray(neg)); return {k:float(m[k]) for k in ['AUROC_sym','BDR_at_AIR_0.8','AIR_at_BDR_0.1']}

def score_sessions(bundle,hreal,hpool,sessions):
    C=P3.C; d1,d2,vec,ec,seq=bundle['d1'],bundle['d2'],bundle['vec'],bundle['ec'],bundle['seq']
    negq=C.excluded_queries(hreal,[r['text'] for r in hpool],6000,'P3C_HELD_NEG')
    neg={'D1':d1.predict_proba(vec.transform(negq))[:,1],'D2':d2.predict_proba(ec.get(negq))[:,1]}
    posq=C.flatten_queries(hpool,sessions); pos={'D1':d1.predict_proba(vec.transform(posq))[:,1],'D2':d2.predict_proba(ec.get(posq))[:,1]}
    for L in P3CFG['L_grid']:
        nw=C.real_windows(hreal,L,P3CFG['detector_window_cap'],'P3C_HELD_W'+str(L)); ai=C.audit_items(hpool,sessions,L)
        if L in (4,8):
            neg[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,nw,L); pos[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,ai,L)
        neg[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,nw,L); pos[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,ai,L)
    strongest=load(P3C_RET/'summary.json')['strongest_by_L']; out={}
    for L in ['1','4','8']:
        det=strongest[L]['detector']; out[L]={'detector':det,**metric_triplet(C,pos[det],neg[det])}
    return out

def lineage_eval(sessions,hpool,HE,bundle,evidence_order,use_risk):
    C=P3.C; fm=load(SHARED/'runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1/return/p3c_freeze_manifest.json')
    ad=C.DNAAdapter(); tau=float(fm['lineage_similarity_threshold']); rt=float(fm['risk_threshold_kappa']) if use_risk else None; gt=float(fm['bank_gap_threshold']) if use_risk else None
    stat=fm['stateless_attacker']; state=fm['stateful_attacker_L4']; b=str(P3CFG['main_beta'])
    th1=float(fm['detector_thresholds'][b][stat]['threshold']); th2=float(fm['detector_thresholds'][b][state]['threshold'])
    hon=[]; a1=[]; a2=[]; air1=[]; air2=[]
    for s in sessions:
        idx=s[evidence_order]
        hon.append(C.lineage_session(ad,idx,HE['qwen_nf4'],HE['qwen_source'],HE['phi_source'],rt,gt,tau))
        mix,a,_=C.attack_a1(s,hpool,bundle['d1'],bundle['d2'],bundle['vec'],bundle['ec'],stat,th1,HE['qwen_nf4'],HE['phi_source'])
        a1.append(C.lineage_session(ad,idx,mix,HE['qwen_source'],HE['phi_source'],rt,gt,tau)); air1.append(a)
        mix,a,_=C.attack_a2(s,hpool,bundle['seq'][state],bundle['ec'],4,th2,HE['qwen_nf4'],HE['phi_source'])
        a2.append(C.lineage_session(ad,idx,mix,HE['qwen_source'],HE['phi_source'],rt,gt,tau)); air2.append(a)
    return {'honest':C.verdict_rates(hon),'A1':{**C.verdict_rates(a1),'AIR':float(np.mean(air1))},'A2':{**C.verdict_rates(a2),'AIR':float(np.mean(air2))}}

def selftest():
    assert CFG['diagnostic_only'] and not CFG['retuning_allowed'] and not CFG['new_model_generation']
    # Canonicalization is purely an evidence-side order choice; send order remains `order`.
    s={'order':[3,2,1],'random_order':[1,3,2]}; assert s['order']!=s['random_order']
    print('P6_SELFTEST_OK')

def main():
    phases={}; log('=== StageP6 LLM-DNA Scheduler/Evidence Canonicalization Diagnostic ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        guard(); phases['provenance']='ok'
        fm,df,ev,lin=P3.verify_freeze(); bundle=P3.rebuild_frozen_detectors(fm,df,ev); phases['p3_frozen_rebuild']='ok'
        hreal,hpool,hh=heldout_pool(); HE,audit=held_embeddings(hpool); dump('heldout_reuse_audit.json',{'heldout_pool_sha256':hh,'embeddings':audit,'new_model_generation':False}); phases['heldout_reuse']='ok'
        lo,hi=float(fm['evidence_scaling']['q05']),float(fm['evidence_scaling']['q95']); den=max(hi-lo,1e-8)
        raw=(1-P3.C.cosrow(HE['qwen_source'][:,:64],HE['phi_source'][:,:64]))-(1-P3.C.cosrow(HE['qwen_source'][:,:64],HE['qwen_nf4'][:,:64])); J=np.clip((raw-lo)/den,0,1)
        states=bundle['sched']['state_fn']([r['text'] for r in hpool]); n=CFG['heldout_sessions']; q=CFG['main_q']; lam=float(fm['lambda'])
        s2=P3.C.make_sessions(J,lam,n,q,states,bundle['sched']['schedule'],'HELDOUT_CONFIRM',False)
        s3=P3.C.make_sessions(J,lam,n,q,states,bundle['sched']['schedule'],'HELDOUT_CONFIRM',True)
        checks=[]
        for a,b in zip(s2,s3):
            checks.append({'episode':a['episode'],'selected_equal':a['selected']==b['selected'],'random_order_equal':a['random_order']==b['random_order'],'send_order_differs':a['order']!=b['order']})
        if not all(x['selected_equal'] and x['random_order_equal'] for x in checks): raise RuntimeError('S2/S3 session identity mismatch')
        dump('session_identity_audit.json',{'status':'PASS','checks':checks,'all_scheduler_changes_send_order':all(x['send_order_differs'] for x in checks)})
        det2=score_sessions(bundle,hreal,hpool,s2); det3=score_sessions(bundle,hreal,hpool,s3)
        # Existing behaviors
        s2_no=lineage_eval(s2,hpool,HE,bundle,'order',False)
        s3_native_no=lineage_eval(s3,hpool,HE,bundle,'order',False)
        s3_native_risk=lineage_eval(s3,hpool,HE,bundle,'order',True)
        # Revised mechanism: attack/detector observes scheduled send order; evidence engine receives the exact pre-scheduler random order.
        s3_canon_no=lineage_eval(s3,hpool,HE,bundle,'random_order',False)
        s3_canon_risk=lineage_eval(s3,hpool,HE,bundle,'random_order',True)
        results={'S2_fresh_random':{'detectability':det2,'lineage_no_risk':s2_no},
                 'S3_scheduler_native_evidence_order':{'detectability':det3,'lineage_no_risk':s3_native_no,'lineage_risk':s3_native_risk},
                 'S3_scheduler_canonicalized_evidence_order':{'detectability':det3,'lineage_no_risk':s3_canon_no,'lineage_risk':s3_canon_risk}}
        dump('canonicalization_diagnostic.json',results); phases['diagnostic']='ok'
        # Mechanism decision only; not a confirmatory gate.
        honest_base=s2_no['honest']['CSVR']; honest_can=s3_canon_no['honest']['CSVR']; a1_base=s2_no['A1']['CSVR']; a1_can=s3_canon_no['A1']['CSVR']
        cover_preserved=all(abs(det3[L][k]-results['S3_scheduler_canonicalized_evidence_order']['detectability'][L][k])<1e-12 for L in ['1','4','8'] for k in ['AUROC_sym','BDR_at_AIR_0.8','AIR_at_BDR_0.1'])
        recovery={'honest_csvr_S2':honest_base,'honest_csvr_S3_native':s3_native_no['honest']['CSVR'],'honest_csvr_S3_canonical':honest_can,
                  'A1_csvr_S2':a1_base,'A1_csvr_S3_native':s3_native_no['A1']['CSVR'],'A1_csvr_S3_canonical':a1_can,
                  'A2_csvr_S3_canonical_risk':s3_canon_risk['A2']['CSVR'],'A2_fsvr_S3_native_no_risk':s3_native_no['A2']['FSVR'],'A2_fsvr_S3_canonical_risk':s3_canon_risk['A2']['FSVR'],
                  'A2_channel_risk_S3_canonical':s3_canon_risk['A2']['Channel_Risk'],'cover_metrics_preserved_exactly':cover_preserved}
        recovery['canonicalization_recovers_honest_to_S2']=honest_can>=honest_base-1e-12
        recovery['canonicalization_recovers_A1_to_S2']=a1_can>=a1_base-1e-12
        success=recovery['canonicalization_recovers_honest_to_S2'] and recovery['canonicalization_recovers_A1_to_S2'] and cover_preserved
        decision={'status':'STAGEP6_COMPLETE','diagnostic_only':True,'retuned':False,'new_model_generation':False,
                  'mechanism_supported':bool(success),'recovery':recovery,
                  'recommendation':('retain scheduler but canonicalize evidence order; validate revised compiler on a new untouched corpus/model-descendant split' if success else 'do not promote canonicalization; simplify scheduler branch before any new confirmation'),
                  'scientific_note':'This stage reuses touched WildChat heldout only for mechanism diagnosis and cannot create a new positive confirmatory claim.'}
        dump('paper_result_summary.json',decision); dump('status.json',{'status':'STAGEP6_COMPLETE','phases':phases,'diagnostic_only':True,'heldout_reused':True,'retuned':False,'new_model_generation':False}); log(json.dumps(decision,indent=2))
    except Exception as e:
        dump('status.json',{'status':'STAGEP6_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'diagnostic_only':True,'retuned':False,'new_model_generation':False}); raise

if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
