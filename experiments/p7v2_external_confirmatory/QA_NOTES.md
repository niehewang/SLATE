# QA notes

- Mirror-only asset acquisition; no `huggingface.co` or `github.com` in executable shell.
- Exact ShareGPT SHA guard.
- Existing Phi OASST-LoRA checkpoint must be COMPLETE and adapter file hashes are recorded.
- Stateful traffic eligibility guard runs before any model generation; heldout has a separately predeclared evaluability guard after freeze and before heldout response generation.
- Heldout JSONL lines are routed by early `id` regex and not JSON-parsed before freeze.
- Confirmatory detector identities are frozen; no heldout re-selection.
