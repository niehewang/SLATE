# Stage P5 v1.1 — CacheTagRecovery

Engineering-only resume for the failed P5 v1 diagnostic. The prior run completed provenance plus both frozen detector rebuilds and then failed before ablation because P3C persists held-out embedding banks as `held_qwen_source.npz`, `held_qwen_nf4.npz`, and `held_phi_source.npz`, while P5 v1 requested the logical names without the `held_` prefix.

This package changes only that logical-tag to persisted-filename mapping. It does not generate model responses or embeddings, does not retune lambda/detectors/thresholds, and remains post-confirmatory diagnostic only. A hard provenance guard requires the exact registered P5 v1 failure boundary.

Run: `bash SELFTEST.sh && bash RUN_THIS.sh`.
