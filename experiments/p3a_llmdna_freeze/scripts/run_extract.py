#!/usr/bin/env python3
import argparse, inspect, json, math, os, sys, pathlib, traceback
import numpy as np

def dump(p,obj):
    pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w') as f: json.dump(obj,f,indent=2,default=str)

def supported(cls_or_fn, kwargs):
    try:
        sig=inspect.signature(cls_or_fn)
        ps=sig.parameters
        if any(v.kind==inspect.Parameter.VAR_KEYWORD for v in ps.values()): return kwargs
        return {k:v for k,v in kwargs.items() if k in ps}
    except Exception:
        return kwargs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['source','nf4'],required=True)
    ap.add_argument('--repo',required=True)
    ap.add_argument('--qwen',required=True)
    ap.add_argument('--encoder',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    os.chdir(args.repo)
    from llm_dna import DNAExtractionConfig, calc_dna
    kwargs=dict(
      model_name='Qwen/Qwen3-4B-Base', model_path=args.qwen, model_type='auto',
      dataset='rand', probe_set='rand', max_samples=64, extractor_type='text',
      dna_dim=128, reduction_method='random_projection', sentence_encoder=args.encoder,
      encoder_device='cuda:0', pre_agg_embed_dim=64, normalize_embeddings=False,
      max_length=1024, temperature=0.7, top_p=0.9, output_dir=pathlib.Path(args.out),
      save=True, load_in_8bit=False, load_in_4bit=(args.mode=='nf4'),
      no_quantization=(args.mode=='source'), trust_remote_code=True,
      device='auto', gpu_id=0, log_level='INFO', random_seed=42,
      use_chat_template=False,
    )
    cfg=DNAExtractionConfig(**supported(DNAExtractionConfig,kwargs))
    res=calc_dna(cfg)
    v=np.asarray(res.vector,dtype=np.float64).reshape(-1)
    obj={
      'mode':args.mode,'shape':list(v.shape),'finite':bool(np.isfinite(v).all()),
      'norm':float(np.linalg.norm(v)),'vector':v.tolist(),
      'output_path':str(getattr(res,'output_path',None)),
      'summary_path':str(getattr(res,'summary_path',None)),
      'config_used':{k:str(vv) if isinstance(vv,pathlib.Path) else vv for k,vv in supported(DNAExtractionConfig,kwargs).items()},
    }
    dump(pathlib.Path(args.out).parent/f'{args.mode}_result.json',obj)
    print(json.dumps({k:obj[k] for k in ['mode','shape','finite','norm']},indent=2))
    if not obj['finite'] or obj['shape'] != [128]: raise RuntimeError('invalid DNA vector')
if __name__=='__main__': main()
