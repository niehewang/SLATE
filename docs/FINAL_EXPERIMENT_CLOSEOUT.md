# SLATE experiment closeout

## Current status

The experimental program is closed. The repository preserves the final corrected in-domain matched evaluations together with the earlier threat, mechanism, independent-engine, and external-transfer evidence. No held-out result is used for post-hoc rescue or retuning.

## What is established

- Audit-aware selective serving is a real failure mode for black-box lineage auditing.
- AIR–BDR separates audit interception from benign-service diversion.
- In the corrected in-domain matched evaluation, complete SLATE substantially improves history-aware channel cover relative to Unwrapped.
- Scheduling and evidence-order canonicalization are distinct mechanisms for order-sensitive lineage engines.
- Risk-aware outcomes can surface channel risk or rejection instead of forcing some false-source decisions.

## Boundary results

- The in-domain channel-cover gain does not preserve the original honest or bounded-attack correct-source verification criteria.
- Schedule-Only is competitive with Full under the predeclared method-specificity scale, so the cover effect is not attributable solely to the complete mechanism.
- External Family-B transfer remains negative: the wrapper does not provide a universal cross-family improvement, especially when the wrapped lineage engine lacks adequate honest discriminability.

The strongest supported interpretation is therefore serving-path auditability with explicit cover/verification and transfer boundaries, not a universal defense guarantee.

See `docs/CORRECTED_IN_DOMAIN_EVALUATION.md` and `results/corrected_in_domain/` for the final corrected in-domain evidence.
