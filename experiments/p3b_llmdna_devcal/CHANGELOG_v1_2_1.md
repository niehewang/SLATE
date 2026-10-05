# StageP3B v1.2.1 — Asset Path Resume

- Trigger: v1.2 returned `FAILED_BEFORE_FREEZE` with `heldout_touched=false`.
- Root cause: Phi-3 checkpoint was searched under `/home/jx-vmlab/...`, while the successful P2B return records it at `/data/jx-vmlab/SLATE_TDSC_SERVER_SHARED/public_models/Phi-3-mini-4k-instruct`.
- Fix: prefer the P2B-verified `/data` checkpoint, then fall back to `/home` and local HF caches. No online download.
- Scientific protocol unchanged: candidate_dev=512, q=64, same lambda/beta/L grids, same P3A v1.3 guard, same held-out hard guard.
- This is a pre-freeze engineering resume only; it changes no data-dependent choice or threshold.
