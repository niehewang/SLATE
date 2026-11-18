# Experiment lineage map

The repository preserves both the earlier frozen experiment chain and the corrected in-domain matched evaluation that supersedes the old descendant-informed selector.

| Public folder | Role | Status |
|---|---|---|
| `p2a_llmprint_threat` ... `p2d_llmprint_heldout` | threat characterization | historical/frozen |
| `p3a_llmdna_freeze` | lineage-engine alignment | historical/frozen |
| `p3b_llmdna_devcal`, `p3c_llmdna_heldout` | early in-domain protocol | historical; selector superseded |
| `p3d_channel_integrity` | integrity diagnostic | historical/frozen |
| `p4a_srp_devcal`, `p4b_srp_heldout` | independent-engine evidence | historical/frozen |
| `p5_joint_ablation` | component diagnostic | historical/frozen |
| `p6_canonicalization` | evidence-order diagnostic | historical/frozen |
| `p7v2_external_confirmatory` | external transfer/boundary | historical/frozen |
| `corrected_matched_effectiveness` | source-only matched Full vs Unwrapped | current corrected evidence |
| `operating_region_baselines` | fixed lambda/beta grid + simple/random controls | current corrected evidence |

The corrected stages do not erase the earlier chain: they preserve the negative/boundary history while replacing the in-domain query-information boundary and evaluation-unit defects.
