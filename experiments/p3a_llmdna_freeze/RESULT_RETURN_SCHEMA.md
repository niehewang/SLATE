# P3A v1.2 return

Expected return file:
`/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP3A_LLMDNA_OFFICIAL_FREEZE_V1_2.tar.gz`

Core files on success:
- `status.json`
- `official_release_freeze.json`
- `embedding_asset.json`
- `api_introspection.json`
- `source_q64.json`
- `nf4_q64.json`
- `official_distance_q64.json`
- `adapter_preflight.json`
- `p3b_freeze_manifest.json`
- `paper_result_summary.json`
- `summary.json`
- `run.log`

P3A is not a paper-facing results stage. A successful return must say `P3A_READY_FOR_FULL_SLATE`, after which the only next stage is P3B Full-SLATE Dev/Cal Freeze.
