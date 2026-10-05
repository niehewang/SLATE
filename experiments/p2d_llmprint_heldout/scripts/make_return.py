import pathlib,tarfile,shutil
RUN=pathlib.Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2d_llmprint_heldout_confirmatory_v1'
RET=RUN/'return';RET.mkdir(parents=True,exist_ok=True)
if (RUN/'run.log').exists():shutil.copy2(RUN/'run.log',RET/'run.log')
OUT=pathlib.Path('/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP2D_LLMPRINT_HELDOUT_CONFIRMATORY_V1.tar.gz')
with tarfile.open(OUT,'w:gz') as tf:
    for p in sorted(RET.glob('*')):tf.add(p,arcname=p.name)
print(OUT)
