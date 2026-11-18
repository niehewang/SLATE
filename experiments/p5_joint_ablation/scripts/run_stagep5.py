from __future__ import annotations
import json, hashlib, sys, traceback, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
P3CFG=json.load(open(ROOT/'config/p3.json'))
P4CFG=json.load(open(ROOT/'config/p4.json'))
PREV_RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep5_joint_frozen_ablation_claim_boundary_v1'
PREV_RET=PREV_RUN/'return'
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep5_joint_frozen_ablation_claim_boundary_v1_1_cache_tag_recovery'
RET=RUN/'return'; RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
SHARED=Path.home()/'SLATE_TDSC_SERVER_SHARED'
P3C_RET=SHARED/'runs/slate_stagep3c_llmdna_fullslate_heldout_confirmatory_v1/return'
P4B_RET=SHARED/'runs/slate_stagep4b_srp_fullslate_heldout_confirmatory_v1_1_engineering_recovery/return'

sys.path.insert(0,str(ROOT/'scripts'))
import p3diag as P3
import p4diag as P4

# Configure imported frozen-rebuild modules with their original stage configs, but route diagnostics here.
P3.CFG=P3CFG; P3.C.CFG=P3CFG
P3.RET=RET/'p3_rebuild'; P3.RET.mkdir(parents=True,exist_ok=True); P3.LOG=RUN/'p3_rebuild.log'
P3.C.RET=P3.RET; P3.C.LOG=P3.LOG; P3.C.RUN=RUN/'p3_tmp'; P3.C.CACHE=P3.C.RUN/'cache'; P3.C.CACHE.mkdir(parents=True,exist_ok=True)
P4.CFG=P4CFG; P4.C.CFG=P4CFG
P4.RET=RET/'p4_rebuild'; P4.RET.mkdir(parents=True,exist_ok=True); P4.LOG=RUN/'p4_rebuild.log'
P4.C.RET=P4.RET; P4.C.LOG=P4.LOG


def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(s+'\n')

def dump(name,obj):
    p=RET/name; p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=str)

def load(p): return json.load(open(p,encoding='utf-8'))
def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()

def verify_prior_failure():
    sp=PREV_RET/'status.json'
    if not sp.exists():
        raise RuntimeError('missing prior P5 v1 failed-run provenance; refuse cache-tag recovery')
    st=load(sp)
    ph=st.get('phases',{})
    expected_phases={'provenance':'ok','p3_frozen_rebuild':'ok','p4_frozen_rebuild':'ok'}
    if st.get('status')!='STAGEP5_FAILED' or st.get('diagnostic_only') is not True or st.get('retuned') is not False or st.get('new_model_generation') is not False:
        raise RuntimeError('prior P5 status is not the expected clean diagnostic engineering failure')
    if any(ph.get(k)!=v for k,v in expected_phases.items()):
        raise RuntimeError('prior P5 did not reach the registered cache-read failure boundary')
    err=str(st.get('error',''))
    if 'missing P3C held-out embedding cache' not in err or '/embeddings/qwen_source.npz' not in err:
        raise RuntimeError('prior P5 failure is not the registered held_-prefix cache-tag bug')
    out={'status':'PASS','prior_status':st,'recovery_scope':'P3C heldout cache logical-tag -> persisted held_* filename mapping only','retuning_allowed':False,'new_model_generation':False,'diagnostic_only':True}
    dump('engineering_recovery_audit.json',out)
    return out

def guard():
    s3=load(P3C_RET/'status.json'); s4=load(P4B_RET/'status.json')
    if s3.get('status')!='STAGEP3C_COMPLETE' or s3.get('retuned') is not False:
        raise RuntimeError('P3C is not a clean completed confirmatory run')
    if s4.get('status')!='STAGEP4B_COMPLETE' or s4.get('retuned') is not False or s4.get('new_model_generation') is not False:
        raise RuntimeError('P4B recovery is not a clean completed metrics-only recovery')
    h3=load(P3C_RET/'heldout_pool.json'); h4=load(P4B_RET/'heldout_pool.json')
    if h3.get('heldout_sha256')!=h4.get('heldout_sha256'):
        raise RuntimeError('P3C/P4B heldout pool mismatch')
    a4=load(P4B_RET/'engineering_recovery_audit.json')
    if a4.get('status')!='PASS' or a4.get('second_confirmatory_claim') is not False:
        raise RuntimeError('P4B engineering-recovery provenance not clean')
    out={'status':'PASS','P3C_status':s3,'P4B_status':s4,'heldout_pool_sha256':h3['heldout_sha256'],
         'diagnostic_only':True,'retuning_allowed':False,'new_model_generation':False,
         'scope':'post-confirmatory frozen ablation/claim-boundary diagnostic only; cannot create a new confirmatory claim'}
    dump('provenance_guard.json',out); return out

def heldout_pool():
    hreal=P3.heldout_sessions_all(); hpool=P3.C.freeze_pool(hreal,P3CFG['candidate_heldout'],'P3C_HELDOUT_POOL')
    hh=sha('\n'.join(r['id'] for r in hpool))
    exp=load(P3C_RET/'heldout_pool.json')['heldout_sha256']
    if hh!=exp: raise RuntimeError('heldout pool reconstruction mismatch')
    return hreal,hpool,hh

def held_embeddings(hpool):
    E={}; audit={}
    mapping={
        'qwen_source':'held_qwen_source',
        'qwen_nf4':'held_qwen_nf4',
        'phi_source':'held_phi_source',
    }
    for logical_tag, persisted_tag in mapping.items():
        E[logical_tag],audit[logical_tag]=P4.load_p3c_held_embedding(persisted_tag,hpool)
        audit[logical_tag]['logical_tag']=logical_tag
        audit[logical_tag]['persisted_tag']=persisted_tag
    return E,audit

def fixed_topq(core,J,n,q):
    order=np.argsort(np.asarray(J,float))[::-1][:q].astype(int).tolist()
    return [{'episode':e,'seed':0,'selected':order,'random_order':order,'order':order,'mean_J':float(np.mean(np.asarray(J)[order]))} for e in range(n)]

def tilted_fixed(core,J,lam,n,q,seed):
    sel=core.weighted_session(J,lam,q,seed)
    rng=np.random.default_rng(seed+17); order=list(np.asarray(sel)[rng.permutation(len(sel))])
    return [{'episode':e,'seed':int(seed),'selected':list(map(int,sel)),'random_order':list(map(int,order)),'order':list(map(int,order)),'mean_J':float(np.mean(np.asarray(J)[sel]))} for e in range(n)]

def metric_triplet(core,pos,neg):
    m=core.roc_metrics(np.asarray(pos),np.asarray(neg))
    return {k:float(m[k]) for k in ['AUROC_sym','BDR_at_AIR_0.8','AIR_at_BDR_0.1']}

def p3_score_sessions(bundle,hreal,hpool,sessions):
    C=P3.C; d1,d2,vec,ec,seq=bundle['d1'],bundle['d2'],bundle['vec'],bundle['ec'],bundle['seq']
    negq=C.excluded_queries(hreal,[r['text'] for r in hpool],6000,'P3C_HELD_NEG')
    rawneg={'D1':d1.predict_proba(vec.transform(negq))[:,1],'D2':d2.predict_proba(ec.get(negq))[:,1]}
    posq=C.flatten_queries(hpool,sessions); pos={'D1':d1.predict_proba(vec.transform(posq))[:,1],'D2':d2.predict_proba(ec.get(posq))[:,1]}
    for L in P3CFG['L_grid']:
        nw=C.real_windows(hreal,L,P3CFG['detector_window_cap'],'P3C_HELD_W'+str(L)); ai=C.audit_items(hpool,sessions,L)
        if L in (4,8):
            rawneg[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,nw,L); pos[f'D3_L{L}']=C.score_seq(seq[f'D3_L{L}'],ec,ai,L)
        rawneg[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,nw,L); pos[f'D4_L{L}']=C.score_seq(seq[f'D4_L{L}'],ec,ai,L)
    out={}
    fm=load(P3C_RET/'freeze_audit.json') if (P3C_RET/'freeze_audit.json').exists() else None
    frozen=load(SHARED/'runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1/return/p3c_freeze_manifest.json')
    strongest=load(P3C_RET/'summary.json')['strongest_by_L']
    for L in ['1','4','8']:
        det=strongest[L]['detector']; out[L]={'detector':det,**metric_triplet(C,pos[det],rawneg[det])}
    return rawneg,pos,out

def p3_lineage(sessions,hpool,HE,bundle,use_risk):
    C=P3.C; fm=load(SHARED/'runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1/return/p3c_freeze_manifest.json')
    ad=C.DNAAdapter(); tau=float(fm['lineage_similarity_threshold']); rt=float(fm['risk_threshold_kappa']) if use_risk else None; gt=float(fm['bank_gap_threshold']) if use_risk else None
    honest=[C.lineage_session(ad,s['order'],HE['qwen_nf4'],HE['qwen_source'],HE['phi_source'],rt,gt,tau) for s in sessions]
    stat=fm['stateless_attacker']; state=fm['stateful_attacker_L4']; b=str(P3CFG['main_beta'])
    th1=float(fm['detector_thresholds'][b][stat]['threshold']); th2=float(fm['detector_thresholds'][b][state]['threshold'])
    a1=[];a2=[];air1=[];air2=[]
    for s in sessions:
        mix,a,_=C.attack_a1(s,hpool,bundle['d1'],bundle['d2'],bundle['vec'],bundle['ec'],stat,th1,HE['qwen_nf4'],HE['phi_source']); a1.append(C.lineage_session(ad,s['order'],mix,HE['qwen_source'],HE['phi_source'],rt,gt,tau)); air1.append(a)
        mix,a,_=C.attack_a2(s,hpool,bundle['seq'][state],bundle['ec'],4,th2,HE['qwen_nf4'],HE['phi_source']); a2.append(C.lineage_session(ad,s['order'],mix,HE['qwen_source'],HE['phi_source'],rt,gt,tau)); air2.append(a)
    return {'honest':C.verdict_rates(honest),'A1':{**C.verdict_rates(a1),'AIR':float(np.mean(air1))},'A2':{**C.verdict_rates(a2),'AIR':float(np.mean(air2))}}

def p4_score_sessions(bundle,hreal,hpool,sessions):
    neg,pos=P4.score_suite_held(bundle['full'],bundle['ec'],hreal,hpool,sessions,'P5')
    fm=load(SHARED/'runs/slate_stagep4a_srp_fullslate_devcal_freeze_v1/return/p4b_freeze_manifest.json')
    out={}
    for L in ['1','4','8']:
        det=fm['full_detectors']['strongest_by_L'][L]['detector']; key=P4.score_key(det,pos); out[L]={'detector':det,'score_key':key,**metric_triplet(P4.C,pos[key],neg[key])}
    return neg,pos,out

def p4_lineage(sessions,hpool,HE,bundle,rawneg,use_risk):
    fm=load(SHARED/'runs/slate_stagep4a_srp_fullslate_devcal_freeze_v1/return/p4b_freeze_manifest.json')
    tq=float(fm['tau_qwen']); tp=float(fm['tau_phi']); rt=float(fm['risk_threshold_kappa']) if use_risk else None; gt=float(fm['bank_gap_threshold']) if use_risk else None
    hon,att,_,_=P4.attack_eval(sessions,hpool,bundle['full'],bundle['ec'],rawneg,fm['full_detectors'],HE,tq,tp,rt,gt,use_risk)
    return {'honest':hon,'A1':att[str(P4CFG['main_beta'])]['A1'],'A2':att[str(P4CFG['main_beta'])]['A2']}

def selftest():
    assert {'qwen_source':'held_qwen_source','qwen_nf4':'held_qwen_nf4','phi_source':'held_phi_source'}['qwen_source']=='held_qwen_source'
    assert CFG['diagnostic_only'] is True and CFG['retuning_allowed'] is False and CFG['new_model_generation'] is False
    rng=np.random.default_rng(1); J=rng.random(128)
    a=tilted_fixed(P3.C,J,.2,3,64,42); assert len(a)==3 and len(a[0]['order'])==64
    print('P5_SELFTEST_OK')

def main():
    phases={}; log('=== StageP5 Joint Frozen Ablation + Claim Boundary Diagnostic ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        verify_prior_failure(); phases['engineering_recovery_audit']='ok'
        guard(); phases['provenance']='ok'
        # Rebuild both frozen development detector systems without retuning.
        fm3,df3,ev3,lin3=P3.verify_freeze(); b3=P3.rebuild_frozen_detectors(fm3,df3,ev3); phases['p3_frozen_rebuild']='ok'
        fm4,df4,ev4,lin4=P4.verify_freeze(); b4=P4.rebuild_frozen(fm4,df4,ev4); phases['p4_frozen_rebuild']='ok'
        hreal,hpool,hh=heldout_pool(); HE,audit=held_embeddings(hpool); dump('heldout_reuse_audit.json',{'heldout_pool_sha256':hh,'embeddings':audit,'new_model_generation':False}); phases['heldout_reuse']='ok'

        # Frozen held-out evidence scores for each engine.
        lo3,hi3=float(fm3['evidence_scaling']['q05']),float(fm3['evidence_scaling']['q95']); den3=max(hi3-lo3,1e-8)
        raw3=(1-P3.C.cosrow(HE['qwen_source'][:,:64],HE['phi_source'][:,:64]))-(1-P3.C.cosrow(HE['qwen_source'][:,:64],HE['qwen_nf4'][:,:64])); J3=np.clip((raw3-lo3)/den3,0,1)
        lo4,hi4=float(fm4['evidence_scaling']['q05']),float(fm4['evidence_scaling']['q95']); den4=max(hi4-lo4,1e-8)
        J4=np.clip(((1-P4.C.cosrow(HE['qwen_source'],HE['phi_source']))-lo4)/den4,0,1)
        texts=[r['text'] for r in hpool]; st3=b3['sched']['state_fn'](texts); st4=b4['sched']['state_fn'](texts)
        n=CFG['heldout_sessions']; q=CFG['main_q']
        sess3={
          'S0_topq':fixed_topq(P3.C,J3,n,q),
          'S1_tilting_fixed':tilted_fixed(P3.C,J3,float(fm3['lambda']),n,q,P3CFG['seed']+555),
          'S2_fresh_random':P3.C.make_sessions(J3,float(fm3['lambda']),n,q,st3,b3['sched']['schedule'],'HELDOUT_CONFIRM',False),
          'S3_scheduler':P3.C.make_sessions(J3,float(fm3['lambda']),n,q,st3,b3['sched']['schedule'],'HELDOUT_CONFIRM',True)
        }
        sess4={
          'S0_topq':P4.C.fixed_sessions(J4,n,q,'P4B_HELDOUT_UNWRAPPED'),
          'S1_tilting_fixed':tilted_fixed(P4.C,J4,float(fm4['lambda']),n,q,P4CFG['seed']+555),
          'S2_fresh_random':P4.C.make_sessions(J4,float(fm4['lambda']),n,q,st4,b4['sched']['schedule'],'P4B_HELDOUT_FULL',False),
          'S3_scheduler':P4.C.make_sessions(J4,float(fm4['lambda']),n,q,st4,b4['sched']['schedule'],'P4B_HELDOUT_FULL',True)
        }

        p3out={}; p4out={}
        for name,ss in sess3.items():
            neg,pos,det=p3_score_sessions(b3,hreal,hpool,ss); lin=p3_lineage(ss,hpool,HE,b3,False); p3out[name]={'detectability':det,'lineage_no_risk':lin,'mean_J':float(np.mean([s['mean_J'] for s in ss]))}
        # S5 has S3 traffic, with the already-frozen risk-aware verdict.
        neg,pos,det=p3_score_sessions(b3,hreal,hpool,sess3['S3_scheduler']); p3out['S5_full_risk_aware']={'detectability':det,'lineage':p3_lineage(sess3['S3_scheduler'],hpool,HE,b3,True),'mean_J':p3out['S3_scheduler']['mean_J']}

        for name,ss in sess4.items():
            neg,pos,det=p4_score_sessions(b4,hreal,hpool,ss); lin=p4_lineage(ss,hpool,HE,b4,neg,False); p4out[name]={'detectability':det,'lineage_no_risk':lin,'mean_J':float(np.mean([s['mean_J'] for s in ss]))}
        neg,pos,det=p4_score_sessions(b4,hreal,hpool,sess4['S3_scheduler']); p4out['S5_full_risk_aware']={'detectability':det,'lineage':p4_lineage(sess4['S3_scheduler'],hpool,HE,b4,neg,True),'mean_J':p4out['S3_scheduler']['mean_J']}
        dump('ablation_llmdna.json',p3out); dump('ablation_srp.json',p4out); phases['ablation']='ok'

        # Cross-engine scheduler diagnosis uses paired S2->S3 changes and the already-frozen theory bridge.
        t3=load(P3C_RET/'theory_bridge_heldout.json'); t4=load(P4B_RET/'theory_bridge_heldout.json')
        sched={}
        for eng,out,t in [('LLM-DNA',p3out,t3),('SRP',p4out,t4)]:
            s2=out['S2_fresh_random']['detectability']; s3=out['S3_scheduler']['detectability']
            sched[eng]={'L4_AUROC_change_S3_minus_S2':float(s3['4']['AUROC_sym']-s2['4']['AUROC_sym']),
                        'L8_AUROC_change_S3_minus_S2':float(s3['8']['AUROC_sym']-s2['8']['AUROC_sym']),
                        'L4_BDRatAIR08_change_S3_minus_S2':float(s3['4']['BDR_at_AIR_0.8']-s2['4']['BDR_at_AIR_0.8']),
                        'L8_BDRatAIR08_change_S3_minus_S2':float(s3['8']['BDR_at_AIR_0.8']-s2['8']['BDR_at_AIR_0.8']),
                        'theory_delta_sched_improvement_L4':float(t['delta_sched']['4']['improvement']),
                        'theory_delta_sched_improvement_L8':float(t['delta_sched']['8']['improvement'])}
        repeated_negative=all(sched[e]['theory_delta_sched_improvement_L4']<0 and sched[e]['theory_delta_sched_improvement_L8']<0 for e in sched)
        sched['cross_engine_decision']={'repeated_negative_theory_bridge':repeated_negative,
          'recommendation':'demote/remove history scheduler from headline method; do not rescue on touched heldout; any revised method requires a new untouched confirmation source' if repeated_negative else 'retain pending interpretation'}
        dump('scheduler_cross_engine_diagnostic.json',sched)

        p3paper=load(P3C_RET/'paper_result_summary.json'); p4paper=load(P4B_RET/'paper_result_summary.json')
        joint={
          'status':'STAGEP5_COMPLETE','diagnostic_only':True,'retuned':False,'new_model_generation':False,
          'confirmatory_inputs':{'P3C':'complete/frozen','P4B':'complete/frozen engineering recovery'},
          'gate_assessment':{
            'G2':{'LLM-DNA':'fails absolute branch at L8=%.3f; no formal matched-unwrapped branch'%p3paper['L8_auroc'],'SRP':'passes absolute branch','overall':'mixed; narrow cover claim only'},
            'G3':{'SRP':'FAIL (L4 gain %.3f)'%p4paper['gate_decisions']['G3_security_cost']['L4_gain'],'overall':'global two-engine gate FAIL regardless of unresolved LLM-DNA matched baseline'},
            'G4':{'LLM-DNA':'requires same-pool matched unwrapped for formal gate; Full honest CSVR=%.5f'%p3paper['honest_csvr'],'SRP':'mechanical pass is non-informative because both honest CSVR=0','overall':'not usable as broad positive claim'},
            'G5':{'LLM-DNA':'FAIL (A1 %.5f, A2 %.5f vs honest %.5f)'%(p3paper['A1_csvr'],p3paper['A2_csvr'],p3paper['honest_csvr']),'SRP':'FAIL; honest baseline zero','overall':'FAIL'},
            'G6':'FAIL from P3D; private consistency demoted'
          },
          'scheduler':sched,
          'claim_boundary':{
            'keep':['audit-aware selective-serving threat evidence','AIR-BDR formalization','engine-agnostic compiler framing as a design objective','SRP absolute cover result as bounded evidence','negative/boundary evidence'],
            'drop_or_narrow':['universal positive end-to-end defense claim','G3 security-cost improvement across engines','G5 robust verification retention across engines','history scheduler as headline benefit','private consistency as general replay/framing detector'],
            'next_scientific_action':'Use this ablation to decide a simplified revised SLATE. If revised, confirm only on an untouched corpus/model-family split; do not reuse current WildChat heldout as confirmatory.'
          }
        }
        dump('joint_claim_boundary.json',joint); dump('paper_result_summary.json',joint); phases['claim_boundary']='ok'
        dump('status.json',{'status':'STAGEP5_COMPLETE','phases':phases,'diagnostic_only':True,'heldout_reused':True,'retuned':False,'new_model_generation':False})
        log(json.dumps(joint,indent=2))
    except Exception as e:
        dump('status.json',{'status':'STAGEP5_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'diagnostic_only':True,'retuned':False,'new_model_generation':False}); raise

if __name__=='__main__':
    if '--selftest' in sys.argv: selftest()
    else: main()
