# SLATE StageP3B — LLM-DNA Full-SLATE Development/Calibration Freeze

This stage follows `SLATE_TDSC_EXPERIMENT_DIRECTION_20261004_v2`.

Scope: development/calibration only. It MUST NOT read held-out traffic.

Frozen external engine:
- official LLM-DNA/RepTrace 1.0.1 (commit 80dd337... from the StageP3A local freeze)
- `TextDNAExtractor`
- Qwen3-Embedding-8B response encoder
- first 64 response-embedding coordinates
- 128-D shared Gaussian projection, seed 42
- `normalize_embeddings=False`

StageP3A already completed the expensive q=64 source/NF4 official extraction. StageP3A v1.3 has already finalized the `DNASignature.signature` accessor and written `STAGEP3A_COMPLETE`. P3B requires that finalized status before starting and only re-validates the adapter deterministically. It does not rerun StageP3A q=64 generation.

Primary development family:
- source: Qwen3-4B-Base
- honest descendant: NF4 runtime quantization
- unrelated/wrong-source proxy: Phi-3-mini-4k-instruct

The primary candidate benign audit pool is frozen at **512 development queries** from WildChat development, matching the P3B handoff spec. A disjoint 128-query WildChat calibration pool is used only for parameter/threshold freeze evaluation; held-out is never read. Per audit session q=64. No extra cover queries are added.

Query evidence for LLM-DNA is defined in the same frozen response-embedding coordinate system. For query x:
`J_raw(x) = d_cos(source(x), Phi(x)) - d_cos(source(x), NF4(x))`.
The 5th/95th development quantiles map `J_raw` to a clipped [0,1] `J_E`. Calibration uses the development mapping unchanged.

SLATE sampling uses `Q_lambda(x) ∝ P_B(x) exp(J_E(x)/lambda)` with empirical uniform P_B over the frozen benign candidate pool, weighted sampling without replacement, and an ephemeral private seed per session. Candidate lambda grid is frozen before results: {0.05, 0.1, 0.2, 0.5, 1.0}.

History scheduling uses MPNet query embeddings, 8 semantic clusters, 4 length bins, and the development benign transition model. It only reorders the selected q queries.

Red-team detectors are frozen from development and evaluated on calibration: D1 lexical/stylometric, D2 frozen semantic embedding classifier, D3 recurrent L-request detector, D4 stronger bidirectional recurrent detector. L={1,4,8}; beta={.01,.05,.10,.20}; primary beta=.10 and L=4.

The LLM-DNA verification threshold is frozen on calibration negatives only. Risk-aware thresholds are frozen on honest calibration sessions only. Lambda selection is predeclared: among settings with honest risk-aware CSVR >= .95, maximize strongest-L4 BDR@AIR=.8; tie-break by lower strongest-L4 AUROC, then higher mean J_E. If none reaches .95, choose the setting with highest honest CSVR and apply the same tie-breaks.

Paper-facing development fields: Honest/A1/A2 CSVR, FSVR, AIR, BDR, AIR-BDR, BDR@AIR=.8, AIR@BDR=.1, strongest-detector AUROC at L=1/4/8, Channel-Risk, Insufficient-Evidence, mean J_E, Khat_q, Gamma_select, delta_sched,L, epsilon_D,L and bootstrap 95% CIs.

No P3B setting may be changed after StageP3C held-out begins.
