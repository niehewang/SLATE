# StageP2A v1.3 — suffix decode compatibility hotfix

This is an engineering-only resume of the frozen StageP2A protocol.

Scientific settings are unchanged: the same 128 official LLMPrint GCG suffix artifacts, 64/64 train/eval split, D1-D4 architectures, L={1,4,8}, WildChat development/calibration firewall, and BDR budgets are reused.

The only fix is serialization compatibility when decoding official LLMPrint `result.best_ids`: generated tensors may retain a singleton batch dimension such as `[1,n]`, while `tokenizer.decode` expects a flat token-id sequence. v1.3 flattens the already-frozen tensor before decoding. No token is added, removed, optimized, regenerated, or reordered.

No Qwen model or GCG optimization is run in v1.3.
