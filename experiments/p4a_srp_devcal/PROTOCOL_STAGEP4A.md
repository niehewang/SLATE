# P4A frozen protocol

1. Inputs: P3B-complete dev/cal response embedding banks only; WildChat development/calibration only.
2. Hard guard: no held-out split is addressable in this package.
3. SRP: per-query source response embeddings; 10% trimmed mean cosine aggregation.
4. J_E: cosine distance between the two enrolled **source** response prototypes, scaled with development q05/q95. Suspect responses are excluded from J_E.
5. Full-SLATE: same q=64, lambda grid, beta grid, L grid, randomization, scheduler, D1–D4 classes and lambda-choice rule as P3B.
6. Matched unwrapped baseline: deterministic top-q by the same SRP J_E, same query budget, separately trained strongest red-team detectors, same benign BDR budgets.
7. Source thresholds are frozen from cross-source calibration negatives. Risk/gap thresholds are calibration-only.
8. P4B must rebuild frozen detectors and reproduce P4A metrics before first held-out access. No retuning after P4A.
