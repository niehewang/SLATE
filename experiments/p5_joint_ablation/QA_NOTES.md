# QA Notes — P5 v1.1 CacheTagRecovery

The P5 v1 return proved that provenance and P3/P4 frozen rebuilds pass. Failure occurred before ablation while requesting `.../embeddings/qwen_source.npz`. Successful P4B provenance shows the persisted P3C files are `held_qwen_source.npz`, `held_qwen_nf4.npz`, and `held_phi_source.npz`. v1.1 only normalizes these cache tags and records both logical and persisted names in the reuse audit.
