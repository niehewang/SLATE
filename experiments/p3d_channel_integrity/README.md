# Stage P3D — LLM-DNA Channel Integrity

This stage evaluates the already-defined private consistency `risk` statistic. It does not add or tune a detector.

Attacks:
- A2: frozen beta=.10 stateful selective routing from P3C.
- A3-exact: exact-query cached Qwen-source response replay.
- A3-semantic: semantically nearest different-query cached Qwen-source replay within the frozen scheduler state.
- A4: full wrong-source/proxy responses from Phi-3-mini-4k-instruct.

Operating points:
- Existing frozen P3B `risk_threshold_kappa` is reported unchanged.
- A separate evaluation-only TPR@5%FAR point is frozen from development/calibration honest risk scores before heldout scoring; it never changes the main verdict threshold.

G6 is evaluated conservatively: A3 counts as passed only if both exact and semantic variants satisfy AUROC >= .80. The global G6 requires at least two of A2/A3/A4 to satisfy AUROC >= .80 while heldout honest FAR at the frozen P3B kappa is <= .05.

No new model responses are generated. P3C banks/embeddings are reused and hash-checked.
