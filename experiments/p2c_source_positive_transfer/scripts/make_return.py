import os,tarfile
from pathlib import Path
run=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2c_llmprint_source_positive_threat_transfer_v1'
out=Path('/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP2C_LLMPRINT_SOURCE_POSITIVE_THREAT_TRANSFER_V1.tar.gz')
files=['status.json','summary.json','source_positive_results.json','source_replicate_manifest.json','detector_reconstruction.json','frozen_provenance.json']
with tarfile.open(out,'w:gz') as t:
    for n in files:
        p=run/'return'/n
        if p.exists():t.add(p,arcname=n)
    p=run/'run.log'
    if p.exists():t.add(p,arcname='run.log')
print(out)
