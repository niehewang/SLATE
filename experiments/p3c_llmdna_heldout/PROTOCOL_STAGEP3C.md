# Stage P3C — LLM-DNA Full-SLATE Held-out Confirmatory

This package is confirmatory. It must not retune lambda, q, lineage/risk thresholds, detector thresholds, detector architecture, attacker identity, scheduler hyperparameters, generation hyperparameters, or evidence scaling.

## Hard order of operations
1. Verify P3B completed and the semantic hashes of its frozen artifacts match the embedded copies.
2. Reconstruct the frozen D1–D4 detectors using development/calibration only. Audit reconstruction against P3B metrics and frozen calibration thresholds.
3. Only after steps 1–2 pass, set `heldout_touched=true` and read the held-out traffic split.
4. Build a deterministic held-out candidate pool (128, q=64), generate response banks, and evaluate once.
5. Emit paper-facing confirmatory metrics and CIs. Never tune on the held-out results.

The frozen strongest detector identities are D4_L1, D3_L4 and D3_L8, selected in P3B. P3C reports these same detectors; it does not re-select on held-out data.
