#!/usr/bin/env python3
import glob, hashlib, inspect, json, os, pathlib, shutil, subprocess, sys, time, traceback
import numpy as np

RUN=pathlib.Path(os.environ.get('SLATE_P3A_RUN', os.path.expanduser('~/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3a_llmdna_official_freeze_v1_2')))
PKG=pathlib.Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('SLATE_PYTHON','/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python')
QWEN=pathlib.Path('/home/jx-vmlab/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539')
SHARED=pathlib.Path(os.path.expanduser('~/SLATE_TDSC_SERVER_SHARED'))
BASE_REPO=SHARED/'public_baselines_sources'/'LLM-DNA'
REPO=RUN/'frozen_release'/'LLM-DNA-v1.0.1'
PYPI_VENDOR=RUN/'vendor_llmdna_1_0_1'
ENC=SHARED/'public_models'/'Qwen3-Embedding-8B'
P1V=SHARED/'runs'/'slate_stagep1_published_baselines_honest_native_reproduction_v1_1'/'vendor'
RUN.mkdir(parents=True,exist_ok=True)

def write(name,obj):
    p=RUN/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,default=str))

def cmd(c,env=None,cwd=None,timeout=None):
    t=time.time()
    p=subprocess.run(c,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    return {'cmd':c,'rc':p.returncode,'seconds':round(time.time()-t,3),'out':p.stdout[-30000:]}

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def model_complete(p):
    p=pathlib.Path(p)
    if not (p/'config.json').exists(): return False
    weights=list(p.glob('*.safetensors'))+list(p.glob('*.bin'))
    if not weights: return False
    tok=any((p/x).exists() for x in ['tokenizer.json','tokenizer_config.json','vocab.json','tokenizer.model'])
    return tok

def prepare_release():
    """Never contacts GitHub. Prefer local tag archive; fall back to official PyPI artifact."""
    REPO.parent.mkdir(parents=True,exist_ok=True)
    if (REPO/'pyproject.toml').exists() and 'version = "1.0.1"' in (REPO/'pyproject.toml').read_text(errors='ignore'):
        return {'mode':'existing_frozen_copy','python_path':str(REPO/'src'),'repo':str(REPO)}

    if (BASE_REPO/'.git').exists():
        chk=cmd(['git','-C',str(BASE_REPO),'cat-file','-e','v1.0.1^{commit}'],timeout=30)
        if chk['rc']==0:
            commit=cmd(['git','-C',str(BASE_REPO),'rev-list','-n','1','v1.0.1'],timeout=30)['out'].strip().splitlines()[-1]
            REPO.mkdir(parents=True,exist_ok=True)
            # local-only archive: no network, no mutation of the StageP0 clone
            r=subprocess.run(['bash','-lc',f"set -o pipefail; git -C {shlex_quote(str(BASE_REPO))} archive --format=tar v1.0.1 | tar -xf - -C {shlex_quote(str(REPO))}"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if r.returncode==0 and (REPO/'pyproject.toml').exists():
                return {'mode':'local_stagep0_git_archive','python_path':str(REPO/'src'),'repo':str(REPO),'commit':commit,'expected_commit_prefix':'80dd337','expected_match':commit.startswith('80dd337')}
            shutil.rmtree(REPO,ignore_errors=True)

    # Official PyPI fallback. PyPI has been reachable on this server; no GitHub is contacted.
    dldir=RUN/'pypi_release'; dldir.mkdir(parents=True,exist_ok=True)
    dl=cmd([PYTHON,'-m','pip','download','--disable-pip-version-check','--no-deps','--only-binary=:all:','--dest',str(dldir),'llm-dna==1.0.1'],timeout=600)
    wheels=sorted(dldir.glob('llm_dna-1.0.1-*.whl'))
    if dl['rc']!=0 or not wheels:
        raise RuntimeError('Official LLM-DNA v1.0.1 unavailable from both local StageP0 tag and PyPI. PyPI output: '+dl['out'][-3000:])
    wheel=wheels[0]
    shutil.rmtree(PYPI_VENDOR,ignore_errors=True); PYPI_VENDOR.mkdir(parents=True,exist_ok=True)
    ins=cmd([PYTHON,'-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(PYPI_VENDOR),str(wheel)],timeout=600)
    if ins['rc']!=0: raise RuntimeError('PyPI wheel install failed: '+ins['out'][-3000:])
    return {'mode':'official_pypi_wheel','python_path':str(PYPI_VENDOR),'repo':None,'artifact':str(wheel),'artifact_sha256':sha256_file(wheel),'pip_download':dl,'pip_install':ins}

def shlex_quote(s):
    import shlex; return shlex.quote(s)

def find_encoder():
    c=[]
    if model_complete(ENC): c.append(ENC)
    c += [pathlib.Path(x) for x in glob.glob('/home/jx-vmlab/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/*')]
    c += [pathlib.Path(x) for x in glob.glob('/data/**/Qwen3-Embedding-8B',recursive=True)[:20]]
    return next((p for p in c if model_complete(p)),None)

def encoder_download():
    ENC.mkdir(parents=True,exist_ok=True)
    code="""
import os
from huggingface_hub import snapshot_download
p=snapshot_download(repo_id='Qwen/Qwen3-Embedding-8B',local_dir=os.environ['OUT'],endpoint='https://hf-mirror.com',max_workers=1,resume_download=True)
print(p)
"""
    e=os.environ.copy(); e.pop('HF_HUB_OFFLINE',None); e.pop('TRANSFORMERS_OFFLINE',None)
    e['HF_ENDPOINT']='https://hf-mirror.com'; e['OUT']=str(ENC)
    return cmd([PYTHON,'-c',code],env=e,timeout=21600)

def main():
  phases={}
  try:
    envinfo={'python':PYTHON,'python_exists':pathlib.Path(PYTHON).exists(),'qwen':str(QWEN),'qwen_exists':QWEN.exists(),
             'run':str(RUN),'base_repo':str(BASE_REPO),'base_repo_exists':BASE_REPO.exists(),'encoder_target':str(ENC),
             'network_policy':'NO_GITHUB; local StageP0 tag archive first; official PyPI fallback; hf-mirror only for Qwen3-Embedding-8B if missing'}
    envinfo['df']=cmd(['df','-h',str(SHARED)])
    envinfo['runtime']=cmd([PYTHON,'-c','import torch,transformers,sentence_transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, sentence_transformers.__version__)'])
    write('environment.json',envinfo); phases['environment']='ok'
    if not QWEN.exists(): raise RuntimeError('Qwen3-4B Base checkpoint missing')

    rel=prepare_release()
    module_root=pathlib.Path(rel['python_path'])
    penv=os.environ.copy(); penv['PYTHONPATH']=os.pathsep.join([str(module_root),str(P1V),penv.get('PYTHONPATH','')])
    penv['TOKENIZERS_PARALLELISM']='false'; penv['CUDA_VISIBLE_DEVICES']=penv.get('CUDA_VISIBLE_DEVICES','0')
    # verify version without GitHub
    vr=cmd([PYTHON,'-c','import llm_dna,inspect,json; print(json.dumps({"module":llm_dna.__file__,"version":getattr(llm_dna,"__version__",None)}))'],env=penv,timeout=60)
    if vr['rc']!=0: raise RuntimeError('LLM-DNA v1.0.1 import failed: '+vr['out'][-3000:])
    rel['import_probe']=vr
    if rel.get('repo'):
        pp=pathlib.Path(rel['repo'])/'pyproject.toml'; rel['pyproject_has_v1_0_1']=pp.exists() and 'version = "1.0.1"' in pp.read_text(errors='ignore')
    else: rel['pyproject_has_v1_0_1']=True
    write('official_release_freeze.json',rel); phases['release']='ok'

    enc_path=find_encoder(); dl=None
    if enc_path is None:
        dl=encoder_download(); enc_path=find_encoder()
    asset={'path':str(enc_path) if enc_path else None,'complete':bool(enc_path),'download':dl,
           'safetensor_files':len(list(enc_path.glob('*.safetensors'))) if enc_path else 0,
           'size_bytes':sum(x.stat().st_size for x in enc_path.rglob('*') if x.is_file()) if enc_path else 0}
    write('embedding_asset.json',asset)
    if not enc_path: raise RuntimeError('Qwen3-Embedding-8B incomplete after local-cache search and hf-mirror attempt')
    phases['encoder']='ok'

    penv['HF_HUB_OFFLINE']='1'; penv['TRANSFORMERS_OFFLINE']='1'
    introspect="""
import inspect,json,llm_dna
from llm_dna import DNAExtractionConfig, calc_dna, TextDNAExtractor
print(json.dumps({'module':getattr(llm_dna,'__file__',None),'version':getattr(llm_dna,'__version__',None),'DNAExtractionConfig':str(inspect.signature(DNAExtractionConfig)),'calc_dna':str(inspect.signature(calc_dna)),'TextDNAExtractor':str(inspect.signature(TextDNAExtractor)),'extract_dna_from_embeddings':str(inspect.signature(TextDNAExtractor.extract_dna_from_embeddings))},indent=2))
"""
    ir=cmd([PYTHON,'-c',introspect],env=penv,cwd=rel.get('repo') or str(RUN),timeout=120)
    if ir['rc']!=0: raise RuntimeError('API introspection failed: '+ir['out'][-3000:])
    try: api=json.loads(ir['out'][ir['out'].find('{'):])
    except Exception: api={'raw':ir}
    write('api_introspection.json',api); phases['api']='ok'

    for mode in ['source','nf4']:
        od=RUN/f'{mode}_official'; od.mkdir(exist_ok=True)
        rr=cmd([PYTHON,str(PKG/'scripts'/'run_extract.py'),'--mode',mode,'--repo',rel.get('repo') or str(RUN),'--qwen',str(QWEN),'--encoder',str(enc_path),'--out',str(od)],env=penv,cwd=rel.get('repo') or str(RUN),timeout=21600)
        write(f'{mode}_run.json',rr)
        if rr['rc']!=0: raise RuntimeError(f'{mode} official extraction failed: '+rr['out'][-6000:])
        src=od.parent/f'{mode}_result.json'; shutil.copy2(src,RUN/f'{mode}_q64.json')
        phases[mode]='ok'

    s=json.load(open(RUN/'source_q64.json')); q=json.load(open(RUN/'nf4_q64.json'))
    a=np.asarray(s['vector'],float); b=np.asarray(q['vector'],float)
    cos=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    dist={'l2':float(np.linalg.norm(a-b)),'cosine':cos,'cosine_distance':float(1-cos),'same_shape':list(a.shape)==list(b.shape)}
    write('official_distance_q64.json',dist); phases['distance']='ok'

    ar=cmd([PYTHON,str(PKG/'scripts'/'adapter_preflight.py'),str(RUN/'adapter_preflight.json'),str(enc_path)],env=penv,cwd=rel.get('repo') or str(RUN),timeout=2400)
    write('adapter_preflight_run.json',ar)
    if ar['rc']!=0: raise RuntimeError('adapter preflight failed: '+ar['out'][-5000:])
    phases['adapter']='ok'

    # Freeze only the protocol inputs P3B is allowed to tune on development/calibration.
    freeze={
      'stage':'P3A_OFFICIAL_FREEZE','paper_facing_claims':False,
      'official_engine':{'name':'LLM-DNA/RepTrace','version':'1.0.1','version_alignment_note':'Current corrected public release exposing TextDNAExtractor; v0.2.3 was frozen in prior recovery only for provenance and is not used for Full-SLATE.','extractor':'TextDNAExtractor','dna_dim':128,'pre_agg_embed_dim':64,'normalize_embeddings':False,'projection':'shared_gaussian','projection_seed':42,'sentence_encoder':'Qwen/Qwen3-Embedding-8B'},
      'family_A':{'source':'Qwen/Qwen3-4B-Base','descendants_present':['NF4 runtime quantization'],'source_checkpoint':str(QWEN)},
      'family_B_plan':{'source':'microsoft/Phi-3-mini-4k-instruct','status':'asset previously bootstrapped in P2; P3B must verify local checkpoint before use','required_descendants':['runtime quantization','independent LoRA/SFT']},
      'main_q':64,'candidate_pool_target':512,'candidate_pool_split':'WildChat development only in P3B',
      'lambda_candidate_grid':[0.05,0.1,0.2,0.5,1.0],
      'beta_grid':[0.01,0.05,0.10,0.20],'main_beta':0.10,'L_grid':[1,4,8],'main_L':4,
      'traffic_split':{'development':0.60,'calibration':0.20,'heldout':0.20},
      'heldout_access_P3A':False,'heldout_access_P3B':False,
      'JE_contract':'P3B computes query-level evidence from the frozen official response-embedding coordinate system on the development candidate pool; P3A only verifies the adapter and does not select lambda or thresholds.',
      'native_threshold_status':'to be frozen in P3B from development/calibration negatives; no threshold selected in P3A',
      'a0_eligibility_status':'to be evaluated after the P3B native threshold freeze; raw source/NF4 q64 distance is recorded here without a binary claim.'
    }
    write('p3b_freeze_manifest.json',freeze)

    paper={'stage':'P3A','scope':'official-interface freeze only','development_or_confirmatory':'preflight/freeze','paper_facing_claims':False,
           'engine':'LLM-DNA/RepTrace v1.0.1','q':64,'source_vector_valid':True,'nf4_vector_valid':True,'adapter_valid':True,
           'source_nf4_l2':dist['l2'],'source_nf4_cosine_distance':dist['cosine_distance'],
           'next_required_stage':'P3B LLM-DNA Full-SLATE Dev/Cal Freeze','claim_supported':'none; P3A only freezes the official coordinate system'}
    write('paper_result_summary.json',paper)
    summary={'workflow':'SLATE_PUBLISHED_BASELINES_STAGEP3A_LLMDNA_OFFICIAL_FREEZE_V1_2','status':'P3A_READY_FOR_FULL_SLATE',
             'paper_facing_claims':False,'official_release':'1.0.1','release_source_mode':rel['mode'],'encoder':str(enc_path),
             'q':64,'source_ok':True,'nf4_ok':True,'adapter_ok':True,'distance':dist,
             'next':'StageP3B LLM-DNA Full-SLATE Dev/Cal Freeze. No additional P3A/preflight stage.'}
    write('summary.json',summary); write('status.json',{'status':'STAGEP3A_COMPLETE','phases':phases})
  except Exception as e:
    write('status.json',{'status':'STAGEP3A_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases})
    write('summary.json',{'workflow':'SLATE_PUBLISHED_BASELINES_STAGEP3A_LLMDNA_OFFICIAL_FREEZE_V1_2','status':'FAILED','error':repr(e),'paper_facing_claims':False})
    write('paper_result_summary.json',{'stage':'P3A','development_or_confirmatory':'preflight/freeze','paper_facing_claims':False,'status':'FAILED_BEFORE_PAPER_CLAIM','error':repr(e)})
    raise
if __name__=='__main__': main()
