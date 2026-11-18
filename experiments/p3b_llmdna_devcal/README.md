# Run StageP3B v1.2.1 — Asset Path Resume

This package supersedes v1.2 only because v1.2 failed before any freeze due to a Phi path mismatch. The successful P2B return proves the local Phi checkpoint is at `/path/to/slate_shared/public_models/Phi-3-mini-4k-instruct`.

Scientific settings are unchanged and held-out remains forbidden.

## Run
```bash
cd /path/to/slate_work
tar -xzf SLATE_Server_PublishedBaselines_StageP3B_LLMDNA_FullSLATE_DevCalFreeze_v1_2_1_AssetPathResume.tar.gz
cd SLATE_Server_PublishedBaselines_StageP3B_LLMDNA_FullSLATE_DevCalFreeze_v1_2_1_AssetPathResume
bash ASSET_PREFLIGHT.sh
bash SELFTEST.sh
bash RUN_THIS.sh
```

Expected return:
`/path/to/slate_work/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP3B_LLMDNA_FULLSLATE_DEVCAL_FREEZE_V1_2_1.tar.gz`
