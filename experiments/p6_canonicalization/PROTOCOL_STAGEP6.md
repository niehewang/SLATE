# Stage P6 — LLM-DNA Scheduler / Evidence-Order Canonicalization Diagnostic

This is a **post-confirmatory diagnostic only**. It reuses the already-touched P3C WildChat heldout pool and persisted response embeddings. It cannot create a new confirmatory claim.

## Hypothesis under test
P5 showed that the history scheduler improves real red-team cover metrics, but LLM-DNA uses order-sensitive concatenation and honest CSVR drops from S2 to S3. P6 tests whether the loss is caused by feeding the server-visible scheduled order directly into the evidence engine.

The revised mechanism separates two orders:
1. **Transmission order**: scheduler order, visible to the serving endpoint and attack detector.
2. **Evidence order**: the exact pre-scheduler private random order (`random_order`), restored by query ID after all responses return.

No query, response, detector, threshold, lambda, lineage threshold, or attack definition is changed.

## Comparisons
- S2 fresh-random: random send order / random evidence order.
- S3 native: scheduled send order / scheduled evidence order.
- S3 canonicalized: scheduled send order / restored pre-scheduler evidence order.
- Risk-aware verdict is evaluated with both S3 native and S3 canonicalized evidence order.

## Success criterion
Mechanism support requires:
- canonicalized S3 honest CSVR >= S2 honest CSVR;
- canonicalized S3 A1 CSVR >= S2 A1 CSVR;
- server-visible detector/AIR-BDR metrics remain exactly the S3 scheduled metrics.

A2 is reported diagnostically but is not allowed to redefine the criterion after inspection.
