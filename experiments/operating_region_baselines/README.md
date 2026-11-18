# Operating-region and simple-baseline evaluation

This public stage evaluates the frozen method-strength grid `lambda={0.05,0.10,0.20,0.50,1.00}` and routing-budget grid `beta={0.01,0.05,0.10,0.20}` across Unwrapped, Marginal-Cover, Schedule-Only, and Full conditions. It also implements exact-k AIR-matched random-routing controls for A1/A2 and 2,000-replicate conversation-cluster bootstrap uncertainty.

The confirmatory pool is disjoint from earlier development/calibration and matched-effectiveness held-out data. For privacy minimization, the prior held-out exclusion manifest is not redistributed. `scripts/build_prior_exclusion.py` converts an authorized local prior-heldout manifest into SHA256-only membership hashes, preserving the exclusion rule without publishing raw conversation identifiers.

## Reproduction

1. Edit only path fields in `config/protocol_config.json` or map them to local assets.
2. Keep q, lambda/beta grids, detector/lineage procedures, seed namespaces, and held-out rules unchanged.
3. Run `python scripts/selftest.py`, then invoke `scripts/run_operating_region.py` with the arguments described by `--help`.

`G4_*` strings inside the code are immutable RNG namespace labels from the executed run and are retained for deterministic reproduction.
