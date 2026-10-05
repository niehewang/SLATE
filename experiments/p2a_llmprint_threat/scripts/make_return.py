from pathlib import Path
import tarfile, shutil
run=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2a_llmprint_threat_surface_v1'
ret=run/'return_v1_3'
out=Path('/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP2A_LLMPRINT_THREAT_SURFACE_V1_3.tar.gz')
ret.mkdir(parents=True,exist_ok=True)
# Always include shared run log if it exists.
for p in [run/'run_v1_3.log']:
    if p.exists(): shutil.copy2(p,ret/p.name)
with tarfile.open(out,'w:gz') as tf:
    for p in sorted(ret.iterdir()):
        if p.is_file(): tf.add(p,arcname=p.name)
print(out)
