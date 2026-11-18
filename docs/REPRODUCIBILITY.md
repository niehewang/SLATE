# Reproducibility notes

## Public contents

The repository contains SLATE orchestration/evaluation code, frozen scientific configs, self-tests, and compact aggregate evidence. It excludes model checkpoints, raw conversational corpora, raw response/embedding banks, long logs, and private server archives.

## Corrected in-domain path

For the current in-domain results use:

- `experiments/corrected_matched_effectiveness/`
- `experiments/operating_region_baselines/`

The earlier P3B/P3C directories remain historical provenance and should not be used as the final in-domain query-information protocol.

Public configs replace machine-specific paths with `/path/to/...`. These path substitutions do not change q, lambda/beta grids, model identities, generation settings, thresholds, seed namespaces, or held-out rules. `EXECUTED_CODE_HASHES.json` records the SHA256 hashes of the exact executed source before public sanitization.

The operating-region stage does not redistribute prior-heldout membership. A bundled generator converts an authorized local prior-heldout manifest into a SHA256-only exclusion manifest; the evaluation code performs hashed membership checks, preserving the exclusion semantics.

## Validation

```bash
python scripts/release_audit.py
python scripts/repo_check.py
python -m compileall -q experiments scripts
python experiments/corrected_matched_effectiveness/scripts/selftest.py
python experiments/operating_region_baselines/scripts/selftest.py
```
