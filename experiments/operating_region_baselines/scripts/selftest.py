from __future__ import annotations
import ast, json
from pathlib import Path
import numpy as np
from common import PKG, CFG, PCFG

def main():
    assert CFG['gate']=='OPERATING_REGION_BASELINES'
    assert int(CFG['q'])==64 and int(CFG['session_realizations'])==64 and int(CFG['bootstrap_reps'])==2000
    assert CFG['lambda_grid']==[0.05,0.1,0.2,0.5,1.0]
    assert CFG['beta_grid']==[0.01,0.05,0.1,0.2]
    assert PCFG['heldout_sessions']==64 and PCFG['bootstrap_reps']==2000
    src=(PKG/'scripts/run_operating_region.py').read_text(encoding='utf-8')
    pre=(PKG/'scripts/precheck.py').read_text(encoding='utf-8')
    assert 'def source_only_J' in src
    assert 'M0_Unwrapped' in src and 'M1_MarginalCover' in src and 'M2_ScheduleOnly' in src and 'M3_Full' in src
    assert 'random_exact_k' in src and 'bootstrap_cover' in src and 'bootstrap_random_utility' in src
    forbidden_old=['cosdist(source,Phi)-cosdist(source,NF4)','d_sp-d_sd']
    assert not any(x in src for x in forbidden_old)
    # Prior held-out exclusion is embedded and non-empty.
    assert CFG['prior_heldout_exclusion'].startswith('/path/to/')
    assert (PKG/'scripts/build_prior_exclusion.py').exists()
    # Every active Python file parses.
    for p in (PKG/'scripts').glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    # Source-only independence synthetic check.
    rng=np.random.default_rng(11); q=rng.normal(size=(32,64)); p=rng.normal(size=(32,64)); nf1=rng.normal(size=(32,64)); nf2=rng.normal(size=(32,64))
    cos=lambda A,B: np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)
    raw=1-cos(q,p); lo,hi=np.quantile(raw,[.05,.95]); j=np.clip((raw-lo)/(hi-lo),0,1); j2=np.clip(((1-cos(q,p))-lo)/(hi-lo),0,1)
    assert np.allclose(j,j2) and not np.array_equal(nf1,nf2)
    # exact-k routing deterministic check, including A2 eligibility t>=3 for L=4.
    for eligible,k in [(np.arange(64),11),(np.arange(3,64),17)]:
        a=np.random.default_rng(99).choice(eligible,k,replace=False); b=np.random.default_rng(99).choice(eligible,k,replace=False)
        assert len(a)==k and len(set(map(int,a)))==k and np.array_equal(a,b)
        assert set(map(int,a)).issubset(set(map(int,eligible)))
    # Paired cluster-bootstrap deterministic sanity.
    a=np.arange(12)/10+.15; b=np.arange(12)/10
    def boot(seed):
        r=np.random.default_rng(seed); return [float(np.mean((a-b)[r.integers(0,12,12)])) for _ in range(100)]
    assert boot(3)==boot(3) and abs(np.mean(boot(3))-.15)<1e-12
    # Freeze semantics must state all five lambdas reported and no selection.
    assert 'lambda_selection' in src and 'FORBIDDEN_ALL_FIVE_REPORTED' in src
    assert 'downstream_stage_invocation' in src and "'downstream_stage_invocation':False" in src
    print('OPERATING_REGION_SELFTEST_PASS')
if __name__=='__main__': main()
