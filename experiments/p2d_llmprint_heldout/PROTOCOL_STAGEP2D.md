# SLATE StageP2D — LLMPrint Held-out Confirmatory v1

## Purpose
Confirm the previously frozen LLMPrint audit-query distinguishability result on **unseen fingerprint prompts and WildChat held-out traffic**. This is a paper-facing confirmatory stage for query/session exposure only. It does not retune LLMPrint, D1–D4, attack thresholds, model families, or any SLATE component.

## Frozen inputs
- P2A D1–D4 architectures, seeds, training split, calibration thresholds, and strongest-detector identities.
- P2A GCG settings: 100 steps, search width 64, top-k 64, prefix cache disabled.
- q=64 and L={1,4,8}.
- BDR budgets beta={0.01,0.05,0.10,0.20}.

## Unseen audit bank
A new official LLMPrint candidate-pair file is generated with seed 20261005. Exact or reversed word-pair overlaps with the original 128-pair P2A bank are removed. The first 64 remaining pairs are frozen **without using any detector score**. Official GCG is then run for these 64 pairs. Exact prompt-text overlap with the original 128 prompts is prohibited.

## Data firewall
1. Generate and freeze the new 64-prompt audit bank.
2. Reconstruct D1–D4 only from the original P2A train bank + WildChat development split.
3. Verify reconstructed detectors on the original P2A eval bank + WildChat calibration split. Frozen calibration BDR must match within 0.02.
4. Write `preheldout_freeze.json` containing the new bank hash and all reconstruction checks.
5. **Only after steps 1–4 pass**, read WildChat held-out traffic.

No threshold, detector, prompt, session schedule, or hyperparameter may change after held-out access.

## Confirmatory outputs
- D1/D2/D3/D4 AUROC on new audit prompts/sessions vs held-out benign traffic.
- Frozen strongest detector at L=1,4,8; identities are not reselected on held-out.
- BDR@AIR=.8 and AIR@BDR=.1.
- Transfer of the frozen P2A thresholds: held-out BDR and unseen-prompt AIR at beta={.01,.05,.10,.20}.
- Bootstrap 95% CI for AUROC and frozen-threshold AIR/BDR.

## Interpretation
This stage can confirm or falsify transfer of the LLMPrint audit-exposure result to unseen prompts/sessions. Negative held-out results are retained; no post-hoc rescue is allowed.
