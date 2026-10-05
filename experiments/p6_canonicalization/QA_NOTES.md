# QA
- diagnostic_only=true
- new_model_generation=false
- retuning_allowed=false
- hard provenance guard requires clean P3C + completed P5
- P3B frozen detector hashes are re-audited before diagnostic
- P3C heldout pool hash is rechecked
- persisted `held_*` embeddings only
- scheduler transmission order is unchanged; only auditor-side evidence ordering differs
