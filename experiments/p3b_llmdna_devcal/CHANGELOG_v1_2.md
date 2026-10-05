# StageP3B v1.2 — Pre-run Safety / Protocol Conformance

Issued before any formal P3B return was present in the project results folder.

Corrections relative to v1.1:
1. Primary development candidate pool increased from 256 to the P3B-spec target **512**. The 128-query calibration pool stays disjoint and is used for development/calibration freeze evaluation.
2. Response generation resumes on stable original batch boundaries, so interruption no longer shifts RNG seeds for later records.
3. `RUN_THIS.sh` always creates a compact return tarball on success **or failure**.
4. Added a hard guard forbidding `heldout` access in P3B.
5. Phi-3 is resolved only from already-local assets (shared model directory or local HF cache); no download is attempted.

Unchanged: LLM-DNA 1.0.1 coordinate, q=64, lambda grid, beta grid, L grid, scheduler design, detector family, risk-aware verdict, and all held-out isolation rules.
