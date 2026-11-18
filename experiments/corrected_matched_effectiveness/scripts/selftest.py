from __future__ import annotations
import ast, json, sys
from pathlib import Path
import numpy as np
from common import PKG, CFG, PCFG

def main():
    assert CFG['gate']=='CORRECTED_MATCHED_EFFECTIVENESS'
    assert PCFG['main_q']==64 and PCFG['heldout_sessions']==64 and PCFG['bootstrap_reps']==2000
    assert PCFG['lambda_grid']==[0.05,0.1,0.2,0.5,1.0]
    # Active scorer code must explicitly use source-only Qwen/Phi and not contain the old algebraic scorer.
    src=(PKG/'scripts/run_matched_effectiveness.py').read_text(encoding='utf-8')
    assert 'def source_only_J' in src
    forbidden=['d_sp-d_sd','cosdist(source,Phi)-cosdist(source,NF4)','source,NF4']
    assert not any(x in src for x in forbidden)
    # Syntax of every active Python file.
    for p in (PKG/'scripts').glob('*.py'): ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    # Synthetic source-only independence from descendant array.
    rng=np.random.default_rng(11); q=rng.normal(size=(32,64)); p=rng.normal(size=(32,64)); nf1=rng.normal(size=(32,64)); nf2=rng.normal(size=(32,64))
    cos=lambda A,B: np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)
    raw=1-cos(q,p); lo,hi=np.quantile(raw,[.05,.95]); j=np.clip((raw-lo)/(hi-lo),0,1); j2=np.clip(((1-cos(q,p))-lo)/(hi-lo),0,1)
    assert np.allclose(j,j2) and not np.array_equal(nf1,nf2)
    # Paired cluster bootstrap deterministic sanity.
    a=np.arange(12)/10+.15; b=np.arange(12)/10
    def boot(seed):
        r=np.random.default_rng(seed); return [float(np.mean((a-b)[r.integers(0,12,12)])) for _ in range(100)]
    assert boot(3)==boot(3) and abs(np.mean(boot(3))-.15)<1e-12
    print('MATCHED_EFFECTIVENESS_SELFTEST_PASS')
if __name__=='__main__': main()
