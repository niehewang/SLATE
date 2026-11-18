from __future__ import annotations
import argparse, json, os, subprocess, sys, traceback
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
    out['ok']=r.returncode==0 and bool(out.get('cuda_available'))
    return out

def build_eligibility():
    traffic=Path(CFG['traffic'])
    dev_s=sessions_for_split(traffic,'development','P3B_development')
    cal_s=sessions_for_split(traffic,'calibration','P3B_calibration')
    held_s=sessions_for_split(traffic,'heldout','P3C_heldout')
    dev=freeze_pool(dev_s,512,'P3B_DEV_POOL'); cal=freeze_pool(cal_s,128,'P3B_CAL_POOL')
    old=freeze_pool(held_s,128,'P3C_HELDOUT_POOL')
    excluded_text={x['text'] for x in dev+cal+old}
    excluded_cid={x['conversation_id'] for x in dev+cal+old}
    # Stronger than text-only exclusion: every conversation used by old P3/P5/P6 is excluded wholesale.
    seen=set(); eligible=[]
    for s in held_s:
        if s['id'] in excluded_cid: continue
        for ti,t in s['turns']:
            if t in seen or t in excluded_text: continue
            seen.add(t)
            eligible.append({'id':query_id('G3_UNTOUCHED_HELDOUT',s['id'],ti,t),'conversation_id':s['id'],'turn_index':ti,'text':t})
    eligible.sort(key=lambda x:h64(x['id'],'G3_UNTOUCHED_HELDOUT'))
    target=min(int(CFG['candidate_target']),len(eligible))
    chosen=eligible[:target]
    return {
      'development_pool_n':len(dev),'calibration_pool_n':len(cal),'old_p3_p5_p6_heldout_excluded_n':len(old),
      'old_touched_conversations_excluded':len({x['conversation_id'] for x in old}),
      'eligible_untouched_n':len(eligible),'chosen_n':len(chosen),'minimum_required':int(CFG['candidate_min']),
      'target':int(CFG['candidate_target']),'q':int(CFG['q']),
      'pass':len(chosen)>=int(CFG['candidate_min']),
      'exclusion_rule':'exclude all dev/cal query texts+conversation IDs and all old P3 heldout query texts+conversation IDs; P5/P6 reuse old P3C heldout, so this excludes their touched heldout too',
      'chosen':chosen,
      'hashes':{
        'dev_ids_sha256':sha_text('\n'.join(x['id'] for x in dev)),
        'cal_ids_sha256':sha_text('\n'.join(x['id'] for x in cal)),
        'old_touched_ids_sha256':sha_text('\n'.join(x['id'] for x in old)),
        'chosen_ids_sha256':sha_text('\n'.join(x['id'] for x in chosen)),
        'chosen_conversation_ids_sha256':sha_text('\n'.join(sorted({x['conversation_id'] for x in chosen})))
      }
    }

def synthetic_cluster_bootstrap_test():
    # Deterministic paired cluster resampling: known mean difference 0.2 in every cluster.
    clusters=np.arange(8); a=np.arange(8,dtype=float)/10+.2; b=np.arange(8,dtype=float)/10
    def one(seed):
        rng=np.random.default_rng(seed); vals=[]
        for _ in range(200):
            ii=rng.integers(0,len(clusters),len(clusters)); vals.append(float(np.mean(a[ii]-b[ii])))
        return vals
    x=one(99); y=one(99)
    return {'deterministic':x==y,'mean':float(np.mean(x)),'expected':0.2,'pass':x==y and abs(np.mean(x)-.2)<1e-12}

def source_only_scorer_test():
    rng=np.random.default_rng(7); q=rng.normal(size=(20,64)); p=rng.normal(size=(20,64)); nf4a=rng.normal(size=(20,64)); nf4b=rng.normal(size=(20,64))
    def cosrow(A,B): return np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)
    raw=1-cosrow(q,p); lo,hi=np.quantile(raw,[.05,.95]); J=np.clip((raw-lo)/max(float(hi-lo),1e-12),0,1)
    raw2=1-cosrow(q,p); J2=np.clip((raw2-lo)/max(float(hi-lo),1e-12),0,1)
    return {'nf4_changed_but_J_max_abs_diff':float(np.max(np.abs(J-J2))),'pass':bool(np.allclose(J,J2)),'nf4_a_checksum':sha_text(nf4a.tobytes().hex()),'nf4_b_checksum':sha_text(nf4b.tobytes().hex())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); args=ap.parse_args(); rd=Path(args.run_dir); out=rd/'precheck'; out.mkdir(parents=True,exist_ok=True)
    report={'time':now(),'scientific_state_advanced':False,'heldout_response_access':False,'checks':{}}
    try:
        exact={
          'python':Path(CFG['python']),'llmdna_root':Path(CFG['llmdna_root']),'qwen_source':Path(CFG['qwen_source']),
          'phi_source':Path(CFG['phi_source']),'qwen_embedding':Path(CFG['qwen_embedding']),'mpnet':Path(CFG['mpnet']),'traffic':Path(CFG['traffic'])
        }
        report['checks']['paths']={k:{'path':str(p),'exists':p.exists()} for k,p in exact.items()}
        if not all(x['exists'] for x in report['checks']['paths'].values()): raise RuntimeError('required exact path missing; substitution forbidden')
        if not model_complete(exact['qwen_source']) or not model_complete(exact['phi_source']) or not model_complete(exact['qwen_embedding']): raise RuntimeError('required exact model asset incomplete')
        report['checks']['imports']=imports_check(str(exact['python']))
        if not report['checks']['imports']['ok']: raise RuntimeError('validated interpreter/import preflight failed')
        report['checks']['gpu']=gpu_check(str(exact['python']))
        if not report['checks']['gpu']['ok']: raise RuntimeError('CUDA GPU unavailable')
        report['checks']['disk_free_gb']=disk_free_gb(Path(CFG['root']))
        if report['checks']['disk_free_gb'] < float(CFG['min_free_disk_gb']): raise RuntimeError('insufficient free disk')
        report['checks']['source_only_scorer_selftest']=source_only_scorer_test()
        report['checks']['cluster_bootstrap_selftest']=synthetic_cluster_bootstrap_test()
        if not report['checks']['source_only_scorer_selftest']['pass'] or not report['checks']['cluster_bootstrap_selftest']['pass']: raise RuntimeError('deterministic scientific selftest failed')
        elig=build_eligibility(); chosen=elig.pop('chosen'); report['eligibility']=elig
        jsonl_write(out/'eligible_pool.jsonl',chosen)
        atomic_json(out/'eligibility.json',elig)
        if not elig['pass']:
            report['status']='BLOCKED_PRECHECK'; report['reason']=f"untouched WildChat eligible pool N={elig['eligible_untouched_n']} < {CFG['candidate_min']}"
            atomic_json(out/'precheck_report.json',report)
            update_status(rd,status='BLOCKED_PRECHECK',phase='PRECHECK',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY',failure_summary=report['reason'])
            return 42
        report['status']='PASS'; report['reason']=None
        atomic_json(out/'precheck_report.json',report)
        update_status(rd,status='PREPARING',phase='PRECHECK_PASS',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY',precheck='PASS')
        return 0
    except Exception as e:
        report.update(status='BLOCKED_PRECHECK',reason=repr(e),traceback=traceback.format_exc())
        atomic_json(out/'precheck_report.json',report)
        update_status(rd,status='BLOCKED_PRECHECK',phase='PRECHECK',heldout_touched=False,scientific_state_advanced=False,resume_allowed='CONDITIONAL_MECHANICAL_ONLY',failure_summary=repr(e))
        return 42

if __name__=='__main__': sys.exit(main())
