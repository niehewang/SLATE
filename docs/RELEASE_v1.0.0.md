# SLATE v1.0.0 — Public Research Code Release

This is the first cleaned public release of the SLATE experiment code.

## Included

- P2 audit-aware selective-serving threat validation;
- P3 LLM-DNA/RepTrace SLATE evaluation chain;
- P4 independent SRP evaluation chain;
- P5 frozen joint ablation / claim-boundary diagnostic;
- P6 evidence-order canonicalization diagnostic;
- P7v2 final ShareGPT + Phi Family-B external confirmation;
- frozen small manifests/configuration files;
- paper-facing experiment closeout and provenance summaries.

## Not included

Model weights, datasets, raw response/embedding banks, checkpoints, long server logs, and full upstream baseline repositories are intentionally excluded.

## Scientific freeze

The main experiment track is closed after the P7v2 one-shot held-out run. Negative and boundary findings are preserved. This release does not retune scientific settings after held-out access.

## Reproduction

Run:

```bash
python scripts/repo_check.py
```

Then consult the per-stage README/protocol and `docs/REPRODUCIBILITY.md`.
