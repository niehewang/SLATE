# SLATE

**SLATE: Serving-Path Lineage Auditability and Traffic Evaluation**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Research Code](https://img.shields.io/badge/research-code-2ea44f.svg)](#)
[![Repository check](https://github.com/niehewang/SLATE/actions/workflows/repo-check.yml/badge.svg)](https://github.com/niehewang/SLATE/actions/workflows/repo-check.yml)
[![Citation](https://img.shields.io/badge/citation-CFF-4c1.svg)](CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code for studying **remote model-lineage verification when a serving endpoint can recognize audit traffic and conditionally alter the serving path**. SLATE treats this as a serving-path auditability problem rather than assuming every query is answered by one fixed backend.

This repository preserves the paper-facing matched evaluations, simple-baseline controls, negative/boundary evidence, and external-transfer results without post-hoc held-out retuning.

## Core question

A remote verifier sees only returned responses, while the provider also observes the query stream and controls request routing. This asymmetry creates three linked questions:

1. how recognizable is audit traffic?
2. how much benign-service diversion is needed to intercept it?
3. how do routing interventions change correct-source verification, false verification, and fail-safe outcomes?

The primary service-path view uses the **AIR–BDR frontier**:

- **AIR — Audit Interception Rate:** fraction of audit traffic intercepted by the serving policy.
- **BDR — Benign Diversion Rate:** fraction of benign traffic diverted by the same policy.

## SLATE mechanisms

The corrected protocol separates:

1. source-only traffic-constrained challenge sampling;
2. history-aware transmission scheduling;
3. evidence-order canonicalization before an order-sensitive lineage extractor;
4. risk-aware final decisions that keep correct verification, false verification, channel risk, insufficient evidence, and rejection distinct.

The final evidence does **not** support a universal lineage-protection claim. In-domain matched experiments show a strong history-aware channel-cover effect, while the original honest and bounded-attack correct-verification criteria are not preserved. Schedule-Only is also competitive with Full under the predeclared method-specificity scale. External transfer remains conditional.

## Repository layout

The public repository separates executable experiments, compact paper-facing results, supporting documentation, and validation scripts. The corrected matched evaluation is under `experiments/corrected_matched_effectiveness/`, and operating-region controls are under `experiments/operating_region_baselines/`. Earlier historical evaluation artifacts are retained only for scientific provenance; the corrected source-only matched evaluation is the authoritative in-domain analysis.

## Quick validation

```bash
python scripts/repo_check.py
python scripts/release_audit.py
python -m compileall -q experiments scripts
python experiments/corrected_matched_effectiveness/scripts/selftest.py
python experiments/operating_region_baselines/scripts/selftest.py
```

## Reproduction boundary

The repository does not redistribute model weights, raw WildChat/ShareGPT/OASST conversations, raw response/embedding banks, long server logs, or private runtime archives. Public stage configs use `/path/to/...` placeholders. Map them to locally obtained assets without changing scientific settings.

For privacy minimization, the corrected operating-region stage requires a local SHA256-only prior-heldout exclusion manifest generated from an authorized local manifest; raw conversation identifiers are not redistributed.

Legacy deterministic RNG namespace labels retained inside the executed experiment code are immutable reproduction identifiers; changing them changes deterministic sampling.

## Data, models, and third-party methods

External evidence/threat methods and model/data assets remain third-party components. The repository contains only SLATE-side orchestration, evaluation logic, compact configs, and aggregate evidence. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Results

Compact corrected in-domain machine-readable evidence is in [`results/corrected_in_domain/`](results/corrected_in_domain/). A concise interpretation is in [`docs/CORRECTED_IN_DOMAIN_EVALUATION.md`](docs/CORRECTED_IN_DOMAIN_EVALUATION.md). Negative and boundary findings are intentionally preserved.

## Citation

If you use this code before the associated paper has a final bibliographic record, please cite the software release using [`CITATION.cff`](CITATION.cff).

## License

Original SLATE repository code is released under the [MIT License](LICENSE). Third-party methods, models, datasets, and assets retain their own licenses and terms.
