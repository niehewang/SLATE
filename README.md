# SLATE

**SLATE: Secure Model-Lineage Auditing against Audit-Aware Selective Serving**

Research code for studying black-box model lineage/source verification when the serving endpoint can detect suspected audit traffic and selectively switch, replay, proxy, or frame responses.

> This public package is a cleaned reproducibility release. It contains experiment code, frozen small manifests/configs, protocols, and QA scripts. It intentionally excludes model weights, private/local datasets, raw response banks, checkpoints, long logs, and server return archives.

## Final scientific scope

The final revised SLATE protocol separates four roles:

1. randomized traffic-constrained challenge sampling;
2. history-aware transmission scheduling;
3. evidence-order canonicalization before order-sensitive lineage extraction;
4. risk-aware abstention as a fail-safe against false verification.

Private consistency / hidden-witness checks are retained only as auxiliary diagnostics, not as a universal A3/A4 headline defense. The final external confirmatory experiment is `p7v2_external_confirmatory`.

## Repository layout

```text
experiments/
  p2a_llmprint_threat/          threat-surface characterization
  p2b_selective_serving/        selective-serving degradation
  p2c_source_positive_transfer/ source-positive threat transfer
  p2d_llmprint_heldout/         held-out threat confirmation
  p3a_llmdna_freeze/            LLM-DNA / RepTrace adapter freeze
  p3b_llmdna_devcal/            Full-SLATE development/calibration
  p3c_llmdna_heldout/           one-shot held-out confirmation
  p3d_channel_integrity/        auxiliary channel-integrity diagnostic
  p4a_srp_devcal/               SRP development/calibration
  p4b_srp_heldout/              SRP held-out confirmation
  p5_joint_ablation/            frozen component ablation
  p6_canonicalization/          evidence-order canonicalization diagnostic
  p7v2_external_confirmatory/   final ShareGPT + Phi Family-B confirmation
docs/
scripts/
```

## Important reproducibility note

The stage directories are the **frozen scripts used by the experiment track**. Some stage scripts therefore contain historical absolute paths from the original offline server. Those paths are not credentials; they document the exact runtime layout. For a new machine, configure equivalent assets using `env.example`, symlinks, or edit only the path-resolution constants before running. Do not change frozen scientific settings (query budget, lambda grid, detector identity, gates, held-out split rules) if you want protocol-equivalent reproduction.

The original server could not access `github.com` or `huggingface.co`; mirror/offline assets were used. The final ShareGPT stage uses a mirror-only asset flow.

## Dependencies

A typical environment requires Python 3.11 and the packages in `requirements.txt`. GPU stages additionally require a CUDA-compatible PyTorch build and, for NF4 paths, a compatible `bitsandbytes` installation.

External research methods are **dependencies**, not claimed as SLATE source code:

- LLMPrint threat-side baseline;
- LLM-DNA / RepTrace evidence engine.

See `THIRD_PARTY.md` and the per-stage protocol files for the frozen versions/adapter assumptions.

## Quick validation

```bash
python scripts/repo_check.py
```

For a specific stage:

```bash
cd experiments/p7v2_external_confirmatory
bash SELFTEST.sh
# then, after preparing the required assets:
bash RUN_THIS.sh
```

## Data and models

This repository does **not** redistribute model checkpoints or conversational corpora. Prepare the required assets separately and respect their upstream licenses. The final external stage expects a fixed ShareGPT asset and local Phi/Qwen/embedding checkpoints; see its `README.md`, `config/project.json`, and `assets/` notes.

## Results and experiment closure

The experiment track was frozen after the final P7v2 one-shot held-out run. See `docs/FINAL_EXPERIMENT_CLOSEOUT.md`. Negative and boundary results are part of the frozen evidence and must not be post-hoc retuned away.

## Citation

Please cite the accompanying SLATE paper once its final bibliographic record is available.

## License

No new repository-wide software license is asserted in this package. Add the license you intend to use before public release, and preserve all upstream third-party license obligations.
