# Legacy stage map

This repository intentionally keeps only the final scientific chain and the final engineering-recovery version when a recovery was required. Earlier failed/hotfix server tarballs are omitted from the public repository.

| Public folder | Experiment lineage | Status |
|---|---|---|
| `p2a_llmprint_threat` | P2A v1.3 | final threat analysis variant |
| `p2b_selective_serving` | P2B | threat degradation |
| `p2c_source_positive_transfer` | P2C | source-positive transfer |
| `p2d_llmprint_heldout` | P2D | held-out confirmation |
| `p3a_llmdna_freeze` | P3A v1.2 (+ finalize-only downstream provenance) | official engine alignment |
| `p3b_llmdna_devcal` | P3B v1.2.1 | final dev/cal freeze |
| `p3c_llmdna_heldout` | P3C | held-out confirmation |
| `p3d_channel_integrity` | P3D | frozen integrity diagnostic |
| `p4a_srp_devcal` | P4A | SRP dev/cal freeze |
| `p4b_srp_heldout` | P4B v1.1 recovery | original one-shot metrics engineering recovery |
| `p5_joint_ablation` | P5 v1.1 recovery | diagnostic-only frozen ablation |
| `p6_canonicalization` | P6 | canonicalization diagnostic |
| `p7v2_external_confirmatory` | P7v2 | final one-shot external confirmation |

OASST P7 attempts are not included as runnable public stages because they terminated pre-heldout on traffic-corpus eligibility; the reason is documented in `docs/P7_TO_P7V2_CORPUS_ELIGIBILITY_PIVOT.md`.
