# Prior held-out exclusion manifest

The operating-region evaluation requires a SHA256-only exclusion manifest derived from the prior matched held-out pool. Raw conversation identifiers are not redistributed.

Expected JSON schema:

```json
{
  "format": "sha256-exclusion-v1",
  "n_queries": 1024,
  "n_conversations": 941,
  "conversation_id_sha256": ["..."],
  "text_sha256": ["..."]
}
```

Generate the file from an authorized local prior-heldout manifest with `scripts/build_prior_exclusion.py`, then set `prior_heldout_exclusion` in `config/protocol_config.json`.
