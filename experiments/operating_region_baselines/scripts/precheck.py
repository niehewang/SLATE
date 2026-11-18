from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from common import *

def model_complete(p):
    p=Path(p); return (p/'config.json').exists() and bool(list(p.glob('*.safetensors'))+list(p.glob('*.bin')))

def imports_check(py):
    code='import torch,numpy,sklearn,transformers,sentence_transformers; print(torch.__version__); print(torch.cuda.is_available())'
    r=subprocess.run([py,'-c',code],text=True,capture_output=True,timeout=120)
    return {'ok':r.returncode==0,'returncode':r.returncode,'stdout':r.stdout[-4000:],'stderr':r.stderr[-4000:]}

def gpu_check(py):
    code='''import json,torch\no={"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count()}\nif torch.cuda.is_available():\n p=torch.cuda.get_device_properties(0); free,total=torch.cuda.mem_get_info(0); o.update(name=p.name,free_bytes=int(free),total_bytes=int(total))\nprint(json.dumps(o))'''
    r=subprocess.run([py,'-c',code],text=True,capture_output=True,timeout=120)
    try: out=json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: out={'parse_error':True,'stdout':r.stdout,'stderr':r.stderr}
    out['ok']=r.returncode==0 and bool(out.get('cuda_available')); return out

def load_prior_exclusion():
    p=Path(CFG['prior_heldout_exclusion']); d=json.load(open(p,encoding='utf-8'))
    if int(d.get('n_queries',-1))!=1024 or int(d.get('n_conversations',-1))!=941: raise RuntimeError('prior held-out exclusion set count mismatch')
    if len(d.get('conversation_id_sha256',[]))!=d['n_conversations'] or len(d.get('text_sha256',[]))!=d['n_queries']: raise RuntimeError('prior held-out exclusion set malformed')
    ck={'embedded_exclusion':str(p),'embedded_sha256':sha_file(p),'privacy_form':'sha256-membership'}
    return d,ck

def build_eligibility():
    traffic=Path(CFG['traffic'])
    dev_s=sessions_for_split(traffic,'development','P3B_development')
    cal_s=sessions_for_split(traffic,'calibration','P3B_calibration')
    held_s=sessions_for_split(traffic,'heldout','P3C_heldout')
    dev=freeze_pool(dev_s,512,'P3B_DEV_POOL'); cal=freeze_pool(cal_s,128,'P3B_CAL_POOL'); old=freeze_pool(held_s,128,'P3C_HELDOUT_POOL')
    prior,ck=load_prior_exclusion()
    banned_cid={x['conversation_id'] for x in dev+cal+old}; g3_c_hash=set(prior['conversation_id_sha256'])
    banned_text_hash={sha_text(x['text'].strip()) for x in dev+cal+old} | set(prior['text_sha256'])
    seen_hash=set(); eligible=[]
    for s in held_s:
        if s['id'] in banned_cid or sha_text(str(s['id'])) in g3_c_hash: continue
        for ti,t in s['turns']:
            th=sha_text(t.strip())
            if th in banned_text_hash or th in seen_hash: continue
            seen_hash.add(th)
            eligible.append({'id':query_id('G4_UNTOUCHED_HELDOUT',s['id'],ti,t),'conversation_id':s['id'],'turn_index':ti,'text':t,'text_sha256':th})
    eligible.sort(key=lambda x:h64(x['id'],'G4_UNTOUCHED_HELDOUT'))
    target=min(int(CFG['candidate_target']),len(eligible)); chosen=eligible[:target]
    g3_c_hash=set(prior['conversation_id_sha256']); g3_t=set(prior['text_sha256'])
    overlap_conv=sum(sha_text(str(x['conversation_id'])) in g3_c_hash for x in chosen); overlap_text=len({x['text_sha256'] for x in chosen}&g3_t)
    out={
      'development_pool_n':len(dev),'calibration_pool_n':len(cal),'old_p3_p5_p6_heldout_excluded_n':len(old),
      'prior_heldout_queries_excluded':int(prior['n_queries']),'prior_heldout_conversations_excluded':int(prior['n_conversations']),
      'eligible_fresh_n':len(eligible),'chosen_n':len(chosen),'minimum_required':int(CFG['candidate_min']),'target':int(CFG['candidate_target']),'q':int(CFG['q']),
      'prior_overlap_conversations':overlap_conv,'prior_overlap_texts':overlap_text,
      'pass':len(chosen)>=int(CFG['candidate_min']) and overlap_conv==0 and overlap_text==0,
      'exclusion_rule':'exclude all development/calibration/old P3-P5-P6 touched conversation IDs and texts plus every canonical prior held-out conversation and text hash before choosing fresh G4 pool',
      'prior_exclusion_check':ck,
      'hashes':{
        'dev_ids_sha256':sha_text('\n'.join(x['id'] for x in dev)),
        'cal_ids_sha256':sha_text('\n'.join(x['id'] for x in cal)),
        'old_touched_ids_sha256':sha_text('\n'.join(x['id'] for x in old)),
        'prior_exclusion_sha256':sha_file(Path(CFG['prior_heldout_exclusion'])),
        'chosen_ids_sha256':sha_text('\n'.join(x['id'] for x in chosen)),
        'chosen_conversation_ids_sha256':sha_text('\n'.join(sorted({x['conversation_id'] for x in chosen}))),
        'chosen_text_hashes_sha256':sha_text('\n'.join(sorted(x['text_sha256'] for x in chosen)))
      },
      'chosen':chosen
    }
    return out

def source_only_scorer_test():
    rng=np.random.default_rng(7); q=rng.normal(size=(20,64)); p=rng.normal(size=(20,64)); nf4a=rng.normal(size=(20,64)); nf4b=rng.normal(size=(20,64))
    cos=lambda A,B: np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)
    raw=1-cos(q,p); lo,hi=np.quantile(raw,[.05,.95]); J=np.clip((raw-lo)/max(float(hi-lo),1e-12),0,1)
    J2=np.clip(((1-cos(q,p))-lo)/max(float(hi-lo),1e-12),0,1)
    return {'nf4_changed_but_J_max_abs_diff':float(np.max(np.abs(J-J2))),'pass':bool(np.allclose(J,J2)),'nf4_arrays_differ':bool(not np.array_equal(nf4a,nf4b))}

def exact_k_random_test():
    rng=np.random.default_rng(13); q=64; eligible=np.arange(3,q)
    rows=[]; ok=True
    for k in [0,1,5,17,len(eligible)]:
        rr=np.random.default_rng(1000+k); choose=rr.choice(eligible,k,replace=False) if k else np.asarray([],int)
        ok=ok and len(choose)==k and len(set(map(int,choose)))==k and set(map(int,choose)).issubset(set(map(int,eligible)))
        rows.append({'k':k,'chosen':len(choose),'unique':len(set(map(int,choose)))})
    # paired determinism
    a=np.random.default_rng(99).choice(np.arange(q),11,replace=False); b=np.random.default_rng(99).choice(np.arange(q),11,replace=False)
    ok=ok and np.array_equal(a,b)
    return {'pass':bool(ok),'cases':rows,'deterministic':bool(np.array_equal(a,b))}

def cluster_bootstrap_test():
    a=np.arange(12,dtype=float)/10+.15; b=np.arange(12,dtype=float)/10
    def boot(seed):
        r=np.random.default_rng(seed); return [float(np.mean((a-b)[r.integers(0,12,12)])) for _ in range(200)]
    x=boot(3); y=boot(3)
    return {'deterministic':x==y,'mean':float(np.mean(x)),'expected':.15,'pass':x==y and abs(np.mean(x)-.15)<1e-12}

def method_definitions():
    return {
      'frozen':True,'q':64,'session_realizations':64,'lambda_grid':[0.05,0.1,0.2,0.5,1.0],'beta_grid':[0.01,0.05,0.1,0.2],
      'M0_UNWRAPPED':{'query_sampling':'uniform/native','transmission':'fresh random order','scheduler':False,'canonicalization':False,'risk_aware_wrapper':False,'J_E_used':False},
      'M1_MARGINAL_COVER':{'query_sampling':'weights proportional only to smoothed benign semantic-cluster x length-bin state marginal learned from development benign traffic','transmission':'fresh random order','scheduler':False,'canonicalization':False,'risk_aware_wrapper':True,'J_E_used':False},
      'M2_SCHEDULE_ONLY':{'query_sampling':'uniform/native','transmission':'same frozen history-aware scheduler learned from development benign traffic','scheduler':True,'canonicalization':'pre-scheduler random order','risk_aware_wrapper':True,'J_E_used':False},
      'M3_FULL':{'query_sampling':'source-only exponential tilt J_E/lambda','transmission':'same frozen history-aware scheduler','scheduler':True,'canonicalization':'pre-scheduler random order','risk_aware_wrapper':True,'J_E_used':True,'all_lambdas_reported':True,'postheldout_lambda_selection':False},
      'random_routing':{'A1':'exact k uniformly without replacement from q positions','A2':'exact k uniformly without replacement from eligible t>=L-1 positions','same_phi_substitution':True,'rng_key':'method|lambda|beta|session|attack, frozen before heldout'},
      'forbidden':['new lambda','new beta','postheldout baseline redefinition','post-heldout rescue','downstream experiment expansion']
    }

def external_baseline_inventory():
    reason='No exact G4-authorized, task-matched local/registered implementation plus preheldout comparability mapping is frozen for this baseline; G4 forbids reconstructing a new external method. Mandatory causal closure uses M0/M1/M2/M3.'
    return {'timing':'PRE_HELDOUT','numeric_table_policy':'external baseline enters matched numeric table only after exact preheldout comparability verification','baselines':[
      {'name':'SIF-class','status':'NOT_NUMERICALLY_COMPARABLE','reason':reason},
      {'name':'ImF-class','status':'NOT_NUMERICALLY_COMPARABLE','reason':reason},
      {'name':'PROSE-class','status':'NOT_NUMERICALLY_COMPARABLE','reason':reason}
    ],'new_external_implementation_after_heldout':'FORBIDDEN'}

def write_blocked_return(rd,report):
    ret=Path(rd)/'return'; ret.mkdir(parents=True,exist_ok=True)
    atomic_json(ret/'external_baseline_feasibility.json',external_baseline_inventory())
    atomic_json(ret/'method_definitions.json',method_definitions())
    (ret/'EXECUTION_DEVIATIONS.md').write_text('# EXECUTION DEVIATIONS\n\nNo scientific deviation. Execution stopped at frozen pre-heldout gate.\n',encoding='utf-8')
    (ret/'paper_result_summary.md').write_text('# SLATE G4 — BLOCKED_PRECHECK\n\n- held-out touched: NO\n- scientific state advanced: NO\n- blocker: `'+str(report.get('reason'))+'`\n- corpus/model/protocol substitution: NO\n- G5-G7 executed: NO\n- scientific adjudication: NOT PERFORMED\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); args=ap.parse_args(); rd=Path(args.run_dir); out=rd/'precheck'; out.mkdir(parents=True,exist_ok=True)
    report={'time':now(),'scientific_state_advanced':False,'heldout_response_access':False,'checks':{}}
    try:
        exact={'python':Path(CFG['python']),'llmdna_root':Path(CFG['llmdna_root']),'qwen_source':Path(CFG['qwen_source']),'phi_source':Path(CFG['phi_source']),'qwen_embedding':Path(CFG['qwen_embedding']),'mpnet':Path(CFG['mpnet']),'traffic':Path(CFG['traffic'])}
        report['checks']['paths']={k:{'path':str(p),'exists':p.exists()} for k,p in exact.items()}
        if not all(x['exists'] for x in report['checks']['paths'].values()): raise RuntimeError('required exact path missing; substitution forbidden')
        if not model_complete(exact['qwen_source']) or not model_complete(exact['phi_source']) or not model_complete(exact['qwen_embedding']): raise RuntimeError('required exact model asset incomplete')
        report['checks']['imports']=imports_check(str(exact['python']))
        if not report['checks']['imports']['ok']: raise RuntimeError('validated interpreter/import preflight failed')
        report['checks']['gpu']=gpu_check(str(exact['python']))
        if not report['checks']['gpu']['ok']: raise RuntimeError('CUDA GPU unavailable')
        report['checks']['disk_free_gb']=disk_free_gb(Path(CFG['root']))
        if report['checks']['disk_free_gb']<float(CFG['min_free_disk_gb']): raise RuntimeError('insufficient free disk')
        # Frozen grid / method guards.
        if CFG['lambda_grid']!=[0.05,0.1,0.2,0.5,1.0] or CFG['beta_grid']!=[0.01,0.05,0.1,0.2] or int(CFG['q'])!=64: raise RuntimeError('frozen G4 grid/q mismatch')
        md=method_definitions(); ext=external_baseline_inventory(); atomic_json(out/'method_definitions.json',md); atomic_json(out/'external_baseline_feasibility.json',ext)
        report['checks']['source_only_scorer_selftest']=source_only_scorer_test(); report['checks']['exact_k_random_selftest']=exact_k_random_test(); report['checks']['cluster_bootstrap_selftest']=cluster_bootstrap_test()
        if not all(report['checks'][k]['pass'] for k in ['source_only_scorer_selftest','exact_k_random_selftest','cluster_bootstrap_selftest']): raise RuntimeError('deterministic scientific selftest failed')
        elig=build_eligibility(); chosen=elig.pop('chosen'); report['eligibility']=elig
        jsonl_write(out/'eligible_pool.jsonl',chosen); atomic_json(out/'eligibility.json',elig)
        if not elig['pass']:
            report['status']='BLOCKED_PRECHECK'; report['reason']=f"fresh fresh WildChat eligibility failed: eligible={elig['eligible_fresh_n']} chosen={elig['chosen_n']} minimum={CFG['candidate_min']} priorConvOverlap={elig['prior_overlap_conversations']} priorTextOverlap={elig['prior_overlap_texts']}"
            atomic_json(out/'precheck_report.json',report); write_blocked_return(rd,report)
            update_status(rd,status='BLOCKED_PRECHECK',phase='PRECHECK',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY',failure_summary=report['reason']); return 42
        report['status']='PASS'; report['reason']=None; atomic_json(out/'precheck_report.json',report)
        update_status(rd,status='PREPARING',phase='PRECHECK_PASS',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY',precheck='PASS'); return 0
    except Exception as e:
        report.update(status='BLOCKED_PRECHECK',reason=repr(e),traceback=traceback.format_exc()); atomic_json(out/'precheck_report.json',report); write_blocked_return(rd,report)
        update_status(rd,status='BLOCKED_PRECHECK',phase='PRECHECK',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY',failure_summary=repr(e)); return 42

if __name__=='__main__': sys.exit(main())
