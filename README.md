# SLATE

**SLATE: Secure Model-Lineage Auditing against Audit-Aware Selective Serving**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Repository check](https://github.com/niehewang/SLATE/actions/workflows/repo-check.yml/badge.svg)](https://github.com/niehewang/SLATE/actions/workflows/repo-check.yml)

Research code for studying **black-box model lineage/source verification when an MLaaS endpoint can detect suspected audit traffic and selectively switch, replay, proxy, or frame responses**.

> **Project status:** experiments are frozen/closed. The accompanying IEEE TDSC manuscript is in preparation. This repository preserves the final experimental code path, including negative and boundary results, without post-hoc retuning.

## Why SLATE?

Most black-box lineage and source-verification methods implicitly assume that every audit query is answered by the same target model. In a hosted API, however, the serving control plane can detect audit-like traffic and conditionally route those requests to a clean model, an incorrect source, a proxy, or cached responses while leaving normal user traffic largely unchanged.

SLATE studies this **audit-aware selective serving** threat and separates the problem into three questions:

1. **Can audit traffic be detected?**
2. **What benign-service disruption is required to intercept it?**
3. **Can the verifier avoid confident false attribution when the serving channel is manipulated?**

The core security view is summarized by the **AIR–BDR frontier**:

- **AIR — Audit Interception Rate:** fraction of audit traffic intercepted by the serving policy.
- **BDR — Benign Diversion Rate:** fraction of benign traffic incorrectly diverted by the same policy.

## Final revised SLATE protocol

The final frozen protocol separates four roles:

1. **Randomized traffic-constrained challenge sampling** — sample informative queries while remaining close to benign traffic.
2. **History-aware transmission scheduling** — shape request order to reduce stateful audit detectability.
3. **Evidence-order canonicalization** — restore a private canonical order before an order-sensitive lineage extractor consumes the responses.
4. **Risk-aware abstention** — surface channel risk / insufficient evidence instead of forcing a false source decision.

Private consistency / hidden-witness checks are retained as **auxiliary diagnostics**, not as a universal A3/A4 defense claim.

## Experimental chain

```text
P2  Threat validation with LLMPrint
    ├─ P2A threat surface
    ├─ P2B selective-serving degradation
    ├─ P2C source-positive threat transfer
    └─ P2D held-out confirmation

P3  LLM-DNA / RepTrace evidence engine
    ├─ P3A official engine freeze
    ├─ P3B SLATE development/calibration
    ├─ P3C one-shot held-out confirmation
    └─ P3D channel-integrity diagnostic

P4  Independent SRP evidence engine
    ├─ P4A development/calibration
    └─ P4B held-out confirmation

P5  Frozen joint ablation / claim-boundary diagnostic
P6  Evidence-order canonicalization diagnostic
P7v2 Final external confirmation: ShareGPT + Phi Family-B
```

See [`LEGACY_STAGE_MAP.md`](LEGACY_STAGE_MAP.md) for the mapping from historical server packages to the public repository.

## Repository layout

```text
experiments/
  p2a_llmprint_threat/
  p2b_selective_serving/
  p2c_source_positive_transfer/
  p2d_llmprint_heldout/
  p3a_llmdna_freeze/
  p3b_llmdna_devcal/
  p3c_llmdna_heldout/
  p3d_channel_integrity/
  p4a_srp_devcal/
  p4b_srp_heldout/
  p5_joint_ablation/
  p6_canonicalization/
  p7v2_external_confirmatory/
docs/
  FINAL_EXPERIMENT_CLOSEOUT.md
  P7V2_MAIN_RESULT_TABLE.md
  P7V2_GATE_SUMMARY.md
  P7V2_PROVENANCE_TABLE.md
  REPRODUCIBILITY.md
scripts/
  repo_check.py
```

## Reproducibility

The stage directories are the **frozen scripts used by the experiment track**. Some scripts intentionally retain historical absolute paths from the original offline server because changing the frozen code after the fact would make the public release diverge from the code that produced the reported results.

For a new machine:

1. create a Python 3.11 environment;
2. install the packages in `requirements.txt` plus a CUDA-compatible PyTorch build where needed;
3. prepare external model/data assets separately;
4. map local asset paths using `env.example`, symlinks, or a protocol-preserving path-resolution edit;
5. do **not** change frozen scientific settings if protocol-equivalent reproduction is required.

The original server could not directly access `github.com` or `huggingface.co`; mirror/offline assets were used. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### Repository sanity check

```bash
python scripts/repo_check.py
```

### Example: final external stage

```bash
cd experiments/p7v2_external_confirmatory
bash SELFTEST.sh
# prepare the required external assets first
bash RUN_THIS.sh
```

## Data, models, and third-party methods

This repository does **not** redistribute model checkpoints, conversational corpora, raw response banks, or upstream research repositories.

External methods such as **LLMPrint** and **LLM-DNA / RepTrace** remain third-party dependencies. Model and dataset assets such as Qwen, Phi, WildChat, ShareGPT, OASST-derived assets, and embedding checkpoints must be obtained separately under their respective licenses and terms.

See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Frozen result status

The final P7v2 held-out experiment closed the main experiment track. The repository intentionally preserves **confirmed, bounded, failed, and non-evaluable findings** rather than retuning after held-out access.

For the paper-facing summary, see:

- [`docs/FINAL_EXPERIMENT_CLOSEOUT.md`](docs/FINAL_EXPERIMENT_CLOSEOUT.md)
- [`docs/P7V2_MAIN_RESULT_TABLE.md`](docs/P7V2_MAIN_RESULT_TABLE.md)
- [`docs/P7V2_GATE_SUMMARY.md`](docs/P7V2_GATE_SUMMARY.md)
- [`docs/P7V2_PROVENANCE_TABLE.md`](docs/P7V2_PROVENANCE_TABLE.md)

## Citation

If you use this code before the paper receives a final bibliographic record, please cite the software release:

```bibtex
@software{nie2026slate,
  author  = {Hewang Nie},
  title   = {SLATE: Secure Model-Lineage Auditing against Audit-Aware Selective Serving},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/niehewang/SLATE}
}
```

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff). After the TDSC paper is published, this section should be updated to the final paper citation/DOI.

## Contributing

Bug reports and reproducibility fixes are welcome. Changes that alter frozen experimental settings should be clearly separated from the paper-reproduction path. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Original SLATE repository code is released under the [MIT License](LICENSE). Third-party methods, models, datasets, and assets remain governed by their own licenses; see [`THIRD_PARTY.md`](THIRD_PARTY.md).
