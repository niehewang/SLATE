# SLATE StageP2C — LLMPrint Source-Positive Threat Transfer v1

## Purpose
StageP2B is frozen as a negative/boundary result: the Qwen3-4B NF4 descendant did not pass the already-frozen LLMPrint native verification threshold under honest serving, so it cannot be used to claim attack-induced CVR degradation. StageP2C does **not** rescue or replace that result. It separately evaluates source-ownership verification, which is a native use case of a fingerprint method, under the same frozen selective-serving setup.

## Frozen inputs
- Same 64 LLMPrint eval fingerprints from StageP2A.
- Same public-validation threshold tau from StageP2B v1.1; no recalibration.
- Same frozen D1-L1 and D3-L4 detector thresholds for beta={.01,.05,.10,.20}.
- Same Phi-3-mini-4k-instruct wrong-model proxy sequence from StageP2B.
- Same original Qwen3-4B base reference sequence from StageP2B.
- WildChat held-out remains untouched.

## Independent honest positive
Generate 3 independent Qwen3-4B source response-sequence replicates on the same 64 frozen fingerprints. Each replicate uses 100 one-token black-box samples per fingerprint with a new fixed seed family. These sequences are **not** the cached reference sequence and therefore measure source verification under independent sampling rather than self-comparison.

## A0 / A1 / A2
A0: independent source responses.
A1: frozen D1-L1 stateless routing; flagged audit requests receive the frozen Phi proxy response distribution.
A2: frozen D3-L4 stateful routing; 100 deterministic session permutations per source replicate; first three requests cannot be flagged because the L=4 window is not yet full.

## No-rescue rule
The StageP2B NF4 boundary remains reported. Tau, query bank, detector thresholds, proxy, q, number of samples, and attack definitions are unchanged. If independent source A0 is unstable or fails the frozen threshold, freeze LLMPrint end-to-end threat transfer as an additional boundary and move on; do not retune.
