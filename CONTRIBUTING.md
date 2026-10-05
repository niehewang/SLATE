# Contributing to SLATE

Thank you for your interest in SLATE.

## Scope

This repository has two layers:

1. **Frozen paper-reproduction code** under `experiments/`.
2. **Repository/documentation tooling** such as README, QA scripts, and reproducibility helpers.

The frozen experiment path is intentionally conservative. Pull requests that change query budgets, lambda grids, detector families/identities, split rules, thresholds, attack definitions, held-out access rules, or gate definitions should not be presented as reproducing the frozen paper experiments.

## Good contributions

- reproducibility fixes that preserve scientific settings;
- clearer error messages and asset checks;
- platform/path portability fixes;
- documentation corrections;
- tests that verify frozen manifests and hashes;
- bug reports with exact commands, environment information, and traceback.

## Before opening a pull request

```bash
python scripts/repo_check.py
python -m compileall -q experiments scripts
```

Do not commit model weights, datasets, response banks, local credentials, access tokens, or private server logs.

## Third-party components

Do not vendor upstream LLMPrint, LLM-DNA/RepTrace, model checkpoints, or datasets into a pull request unless their license explicitly permits redistribution and the provenance/license files are included. See `THIRD_PARTY.md`.
