# SLATE Stage P4A — SRP Full-SLATE Dev/Cal Freeze v1

Purpose: implement the preregistered engine-independent Semantic Response Profile (SRP) and freeze all P4B held-out settings **without accessing held-out traffic**.

Key design decisions:
- Reuses P3B dev/cal response embeddings for Qwen3-4B-Base, its NF4 descendant, and Phi-3-mini-4k-instruct. No new model generation.
- SRP source prototype = per-query frozen response embedding.
- Session score = 10% trimmed-mean cosine similarity to each source prototype.
- Query informativeness J_E uses **source-only** Qwen-vs-Phi prototype separation and never uses suspect/NF4 responses.
- Full-SLATE and a matched unwrapped deterministic top-q SRP baseline are both frozen, so P4B can directly evaluate G2/G3/G4/G5.
- P3D already showed G6 failure; private consistency is not rescued or re-optimized.

Run:
```bash
bash SELFTEST.sh
bash RUN_THIS.sh
```

Expected return:
`/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP4A_SRP_FULLSLATE_DEVCAL_FREEZE_V1.tar.gz`
