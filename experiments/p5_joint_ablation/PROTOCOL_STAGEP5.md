# P5 v1.1 Recovery Protocol

- Purpose: resume the preregistered post-confirmatory frozen ablation after a cache filename mapping bug.
- Allowed code change: `qwen_source -> held_qwen_source`, `qwen_nf4 -> held_qwen_nf4`, `phi_source -> held_phi_source` when reading the already-existing P3C held-out embedding cache.
- Forbidden: new response/embedding generation, detector reselection, threshold changes, lambda/q changes, method rescue, or new confirmatory claims.
- Provenance guard: prior P5 v1 must be `STAGEP5_FAILED`, diagnostic-only, unretuned, no new generation, with provenance/P3/P4 frozen rebuild phases complete and error exactly at the missing unprefixed `qwen_source.npz` cache.
