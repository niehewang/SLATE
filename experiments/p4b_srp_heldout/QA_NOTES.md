Pre-run QA:
- P4A semantic freeze hashes embedded and checked.
- Held-out is inaccessible until detector rebuild audit passes.
- P4B uses P3C_HELDOUT_POOL exactly; no alternate held-out query pool is allowed.
- P3C response embeddings are reused; no model generation path is included.
- Full and unwrapped detector identities/thresholds are frozen from P4A.
- No lambda/scheduler/lineage/risk threshold tuning exists in P4B.
