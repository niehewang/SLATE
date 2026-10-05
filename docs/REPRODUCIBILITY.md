# Reproducibility notes

## What this repository contains

- frozen SLATE orchestration/evaluation code;
- small configuration and freeze manifests;
- self-tests and QA notes;
- paper-facing closeout/provenance summaries.

## What it intentionally excludes

- model checkpoints and adapter weights;
- WildChat, ShareGPT, OASST or other raw corpora;
- raw response/embedding banks;
- large server logs and return archives;
- complete third-party baseline repositories.

## Runtime assumptions

The original experiment server used Python 3.11, CUDA-capable GPUs, local/mirrored model assets, and an offline-style filesystem layout. Some frozen scripts retain those historical absolute paths to preserve provenance.

For a new machine, prefer environment variables, symlinks, or a small path-resolution patch rather than modifying scientific settings.

## Network-constrained servers

The original server could not directly access GitHub or Hugging Face. Assets were obtained through approved mirrors or transferred offline. If reproducing in a similarly restricted environment:

- keep model/data downloads outside the frozen evaluation logic;
- validate asset hashes before running a stage;
- never treat a network failure as a scientific failure;
- do not substitute a different dataset/model silently.

## Frozen-versus-development distinction

P5 and P6 are post-confirmatory diagnostics/mechanism studies and must not be reframed as independent held-out confirmation. P7v2 is the final external one-shot held-out run. The experiment track is closed after P7v2; see `FINAL_EXPERIMENT_CLOSEOUT.md`.

## Sanity checks

```bash
python scripts/repo_check.py
python -m compileall -q experiments scripts
```

Individual stages also provide `SELFTEST.sh` or equivalent preflight scripts.
