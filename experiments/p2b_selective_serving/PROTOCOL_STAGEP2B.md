# SLATE StageP2B — Published LLMPrint Selective-Serving Verification Degradation (Development)

## Role
Paper-facing development experiment after StageP2A. This stage measures whether audit-aware selective serving degrades the *published LLMPrint black-box verification signal*. It is an experiment for the SLATE paper; it is not a network-security benchmark.

## Frozen inputs
- Published method: LLMPrint official prompt-injection fingerprint protocol.
- Fingerprint bank: StageP2A frozen 128 prompts; this stage uses indices 64–127 only (q=64 evaluation bank).
- GCG bank is never regenerated.
- A1 detector: D1 lexical/stylometric from StageP2A.
- A2 detector: D3 sequence detector with L=4 from StageP2A recipe.
- BDR budgets beta in {0.01,0.05,0.10,0.20}; numeric thresholds are copied from StageP2A v1.3.
- WildChat held-out split is never read.

## Models
Reference/source: public Qwen3-4B-Base local checkpoint.
Positive suspect: an independent NF4 4-bit runtime quantized view of Qwen3-4B-Base. This is defined only for SLATE/LLMPrint experiments and carries no FAS genealogy.
Wrong-model proxy: public microsoft/Phi-3-mini-4k-instruct already downloaded by StageP2A. Generation uses use_cache=False only to work around the installed Transformers/Phi remote-code cache API mismatch; this does not change LLMPrint's verification rule. If Phi cannot generate, a public validation-negative proxy is used and the fallback is explicitly recorded.
Validation negatives for the official LLMPrint black-box threshold: small public models from LLMPrint's own released default validation-negative set, prioritized as distilbert/distilgpt2, facebook/opt-350m, bigscience/bloom-560m.

## Native LLMPrint semantics
For each of 64 frozen fingerprints, compose the official prefix + decoded suffix. For each model, draw 100 stochastic one-token samples (temperature=1, top_p=1, top_k=0) and store the frequencies of the two target words. Verification score is official bitwise accuracy against the base reference sequence. The development threshold is the official Gaussian rule tau = mean(validation-negative accuracies) + 1.64 * std, clipped to 1.

## Attacks
A0: honest serving by the quantized suspect.
A1: stateless selective routing. Each fingerprint request is scored by frozen D1; when score >= StageP2A threshold, return proxy-model evidence instead of suspect evidence.
A2: stateful routing at L=4. Across each keyed permutation of the q=64 bank, D3 scores the latest four-query window; the current request is proxied if the score exceeds the frozen StageP2A threshold. First three requests are served honestly. 100 deterministic session permutations are evaluated.

## Outputs
For A0/A1/A2 and each beta: AIR on audit queries, measured calibration BDR, official LLMPrint bitwise accuracy, verification pass rate/CVR, score drop from A0, proxy fraction, and bootstrap intervals. Also returns native base/quantized/proxy/negative sequences, threshold provenance, reconstructed-detector checks, and Phi compatibility status.

## Data firewall
Development and calibration traffic may be used only to reconstruct the already-frozen StageP2A D1/D3 attack policy. Held-out traffic is not opened. No attack threshold is retuned using verification outcomes.

## Stop rule
This is a development measurement. Do not change LLMPrint fingerprints, detector architecture, thresholds, or attack definition in response to these results. Negative results are retained.
