# P3C v1 QA notes

- P3B semantic freeze hashes embedded and checked before confirmatory access.
- Frozen lambda = 0.05, q = 64, beta grid = {0.01,0.05,0.10,0.20}, L = {1,4,8}.
- Frozen strongest detector identities: D4_L1, D3_L4, D3_L8. No held-out re-selection.
- Frozen attacker identities: D1 for A1 and D3_L4 for A2.
- Frozen lineage/risk/bank-gap and detector thresholds are copied from P3B v1.2.1.
- Detector binaries were not persisted by P3B; P3C deterministically rebuilds them from development/calibration and audits AUROC/threshold-transfer against P3B before reading held-out.
- No threshold, lambda, kappa, scheduler hyperparameter, evidence scaling, or generation hyperparameter is fitted on held-out.
- Confirmatory candidate pool is deterministic held-out-only and checked for zero text overlap with development/calibration pools.
