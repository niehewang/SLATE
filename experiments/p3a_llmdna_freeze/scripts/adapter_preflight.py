#!/usr/bin/env python3
import inspect,json,os,pathlib,sys,traceback
import numpy as np

def filt(fn,kw):
    sig=inspect.signature(fn); ps=sig.parameters
    if any(v.kind==inspect.Parameter.VAR_KEYWORD for v in ps.values()): return kw
    return {k:v for k,v in kw.items() if k in ps}

def main():
    out=pathlib.Path(sys.argv[1]); encoder=sys.argv[2]
    from llm_dna import TextDNAExtractor
    init_kw=dict(dna_dim=128,reduction_method='random_projection',sentence_encoder=encoder,
                 pre_agg_embed_dim=64,normalize_embeddings=False,random_seed=42,device='cuda:0',encoder_device='cuda:0')
    sig_init=str(inspect.signature(TextDNAExtractor))
    sig_method=str(inspect.signature(TextDNAExtractor.extract_dna_from_embeddings))
    # Prefer a constructor that does not force-load the 8B encoder for an embeddings-only operation.
    # Try ordinary construction first; if current release exposes a lazy/no-encoder path, signature filtering uses it.
    obj=None; errs=[]
    for trial in [init_kw, {k:v for k,v in init_kw.items() if k!='sentence_encoder'},
                  {k:v for k,v in init_kw.items() if k not in ('sentence_encoder','device')}]:
        try:
            obj=TextDNAExtractor(**filt(TextDNAExtractor,trial)); break
        except Exception as e: errs.append(repr(e))
    if obj is None:
        raise RuntimeError('TextDNAExtractor construction failed: '+ ' | '.join(errs))
    rng=np.random.default_rng(20261004)
    raw=rng.normal(size=(64,1024)).astype(np.float32)
    method=obj.extract_dna_from_embeddings
    def call():
        kw={'model_name':'SLATE_adapter_preflight'}
        try: r=method(raw,**filt(method,kw))
        except TypeError: r=method(raw)
        if hasattr(r,'vector'): r=r.vector
        return np.asarray(r,dtype=np.float64).reshape(-1)
    a=call(); b=call()
    result={'class_signature':sig_init,'method_signature':sig_method,'shape':list(a.shape),
            'finite':bool(np.isfinite(a).all()),'deterministic_max_abs_diff':float(np.max(np.abs(a-b))),
            'norm':float(np.linalg.norm(a))}
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result['shape'] != [128] or not result['finite'] or result['deterministic_max_abs_diff']>1e-7:
        raise RuntimeError('adapter preflight failed')
if __name__=='__main__': main()
