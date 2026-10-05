from pathlib import Path
import tarfile, json
run=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2b_llmprint_selective_serving_degradation_v1'
ret=run/'return'; out=Path('/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP2B_LLMPRINT_SELECTIVE_SERVING_DEGRADATION_V1.tar.gz')
files=['status.json','summary.json','llmprint_native_verification.json','selective_serving_results.json','detector_reconstruction.json','model_sequence_manifest.json','phi_proxy_status.json','validation_negative_status.json']
with tarfile.open(out,'w:gz') as t:
    for f in files:
        p=ret/f
        if p.exists():t.add(p,arcname=f)
    if (run/'run.log').exists():t.add(run/'run.log',arcname='run.log')
print(out)
