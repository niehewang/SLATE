# P4B v1.1 Engineering-Recovery Protocol

- Scientific protocol is identical to frozen P4A/P4B v1.
- Prior held-out access already occurred in P4B v1; this package does **not** constitute a second confirmatory trial.
- Allowed change: normalize frozen L=1 display detector names `D1_L1`/`D2_L1` to the existing score dictionary keys `D1`/`D2` when computing held-out ROC/theory metrics.
- Forbidden: retuning, detector reselection, threshold changes, lambda/q changes, candidate-pool changes, new model generation, or alternate heldout data.
- Recovery must reuse the same P3C heldout pool and embeddings and record `engineering_recovery_audit.json`.
