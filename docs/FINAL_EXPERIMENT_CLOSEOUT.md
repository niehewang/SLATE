# SLATE / IEEE TDSC — FINAL EXPERIMENT CLOSEOUT — 2026-10-05 v1

## Final status

**EXPERIMENT PHASE CLOSED. NO FURTHER RETUNING OR MAIN EXPERIMENT EXPANSION.**

The final blocking experiment, P7v2 external confirmatory evaluation, completed successfully with `heldout_touched=true` and `retuned_after_heldout=false`. Per `SLATE_TDSC_EXPERIMENT_FINALIZATION_GUIDE_20261005_v1`, completion of this one-shot held-out run ends the main experimental program regardless of gate outcome.

## What is established

1. **Threat significance remains established** by the earlier LLMPrint threat-side experiments: audit-aware selective serving can strongly compromise conventional audit traffic at low benign diversion in the Qwen/WildChat setting.
2. **Mechanism diagnosis is established, not externally confirmed as a universal benefit.** P5/P6 showed that transmission scheduling and evidence ordering must be separated for order-sensitive lineage engines; evidence-order canonicalization recovers the scheduler-induced LLM-DNA utility loss on the already-touched WildChat diagnostic data without changing server-visible traffic.
3. **Risk-aware abstention has a defensible fail-safe role.** Across earlier diagnostics and P7v2 LoRA attacks, it can convert suspicious sessions into explicit Inconclusive/Channel-Risk outcomes, but this should not be described as recovery of correct lineage verification.
4. **Underlying lineage-engine eligibility is necessary.** P7v2 LoRA has unwrapped honest CSVR=0, demonstrating that a protocol wrapper cannot manufacture source-lineage discriminability where the wrapped engine lacks it.

## What is not confirmed / what failed

P7v2 independently evaluates the disclosed revised compiler on ShareGPT + Phi-3 Family-B and fails all re-tested positive gates:

- G2: FAIL — revised Full traffic is more detector-separable than matched-unwrapped on held-out L1/L4/L8.
- G3: FAIL — primary L4 BDR@AIR=.8 gain is -0.1164, not >= +0.20.
- G4: FAIL — informative Phi-NF4 honest CSVR drops from 1.0000 unwrapped to 0.7188 Full.
- G5: FAIL — A1/A2 correct-source gains are negative and retention falls below 75% of Full honest.
- G6: not re-tested; private consistency remains demoted after its earlier failure.

Therefore the paper must **not** claim that revised SLATE generally improves cover, attack cost, honest utility, or bounded-attack correct-source retention across independent traffic/model families.

## Final claim boundary for the paper conversation

A defensible paper can still center on the security problem and boundary-aware findings:

- Audit-aware selective serving is a distinct and practically important failure mode for black-box lineage auditing.
- AIR–BDR is a useful way to separate audit interception from benign-service disruption.
- Protocol mechanisms such as randomized traffic shaping, scheduling, canonicalized evidence ordering, and risk-aware abstention expose meaningful design trade-offs.
- The wrapper's benefit is **conditional**, not universal: it depends on the traffic domain and on a lineage engine that already has adequate honest discriminability.
- The external Family-B result should be presented as a negative transfer result, not hidden or rescued.

The strongest manuscript framing is therefore **boundary-aware secure auditing**, not a universal defense guarantee.

## Calibration limitation that must be disclosed

The P7v2 freeze manifest records `unwrapped_honest_cal_mean_csvr=0.0`, so the lambda-selection honest-retention target on calibration was degenerate, while held-out Phi-NF4 has unwrapped honest CSVR=1.0. This development/calibration-to-heldout instability is a material external-domain limitation. Because held-out has been touched, it must be reported rather than used to retune lambda, thresholds, detectors, or risk rules.

## Actions explicitly prohibited after this closeout

- No new model family to obtain a positive result.
- No new traffic corpus to rescue G2/G3/G4/G5.
- No new attack family as a headline experiment.
- No lambda/threshold/detector/risk-rule selection using P7v2 held-out.
- No method revision followed by another main confirmatory run.
- No relabeling of failed gates or changing gate definitions post hoc.

Allowed work is limited to paper-facing aggregation, bootstrap/significance summarization, provenance/hash review, figure/table construction, and correction of presentation-only engineering issues.

## Handoff to the paper conversation

Transfer only the compact paper-facing artifacts in this closeout bundle. Raw checkpoints, raw response banks, and long server logs remain owned by the experiment conversation/Drive archive.

**Final experimental decision:** stop experiments; move the SLATE project to manuscript finalization with confirmed, bounded, and failed claims explicitly separated.
