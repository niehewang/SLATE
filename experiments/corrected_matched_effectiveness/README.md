# Corrected matched-effectiveness evaluation

This public stage contains the corrected in-domain matched comparison between complete SLATE and an unwrapped lineage auditor. The query-information score is source-only: Qwen-source versus Phi-source response embeddings; descendant responses never enter query selection.

The evaluation uses q=64, a fresh WildChat held-out pool, 64 matched session realizations, and conversation-cluster-aware paired bootstrap uncertainty. Thresholds and the method-strength choice are frozen before held-out access.

## Reproduction

1. Edit only the path fields in `config/protocol_config.json` or map them to local assets.
2. Keep all scientific settings unchanged for protocol-equivalent reproduction.
3. Run `python scripts/selftest.py`, then invoke `scripts/run_matched_effectiveness.py` with the arguments described by `--help`.

`G3_*` strings inside the code are immutable RNG namespace labels from the executed run. They are retained because changing them changes deterministic sampling. They are not public workflow gates.

## Provenance

`EXECUTED_CODE_HASHES.json` records SHA256 hashes of the exact executed source. The public files differ only where machine paths, internal workflow metadata, or privacy-sensitive identifiers were converted to portable/public representations.
