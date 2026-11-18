# P4B v1.1 Engineering Recovery

This is not a new confirmatory experiment. It recovers the original frozen P4B one-shot held-out evaluation after a presentation-key mismatch (`D1_L1`/`D2_L1` vs score keys `D1`/`D2`).

Hard guards require the prior P4B v1 run to have failed exactly with `KeyError('D1_L1')` after `heldout_detector_scores`, with heldout touched, no retuning, and the same P3C heldout-pool hash.

No lambda, q, detector identity, detector parameters, lineage thresholds, risk thresholds, attack thresholds, candidate pool, or heldout data are changed. No model response is regenerated.

Run:
```bash
bash SELFTEST.sh
bash RUN_THIS.sh
```
