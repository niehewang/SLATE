# SLATE P7 → P7v2 Corpus-Eligibility Pivot

- The OASST1/Phi P7 attempt stopped before detector freeze and before any heldout access (`heldout_touched=false`, `retuned=false`).
- Model assets, Phi OASST-development LoRA, NF4 runtime setup, dev/cal response banks and suspect-independent evidence were successfully constructed.
- Failure occurred when training the first stateful L=4 detector because the frozen user-side-history extraction produced no valid benign L=4 negative windows. This is a traffic-corpus eligibility failure, not a lineage-model or revised-SLATE result.
- No padding, zero-feature substitution, or cross-conversation stitching is permitted because those would change the stateful threat model.
- P7v2 keeps the revised method frozen after P6 and keeps Phi Family-B; it changes only the external traffic corpus to a multi-turn ShareGPT JSONL corpus.
- Before any P7v2 model inference, hard stateful-eligibility gates require development L4/L8 windows >=512/256 and calibration L4/L8 windows >=128/64. After freeze, one-shot heldout must independently satisfy L4/L8 >=128/64 or be reported non-evaluable.
- The completed Phi LoRA checkpoint trained only on OASST development traffic is reused unchanged and its adapter hashes are recorded.
- P7v2 uses mirror-only asset acquisition and contains no official Hugging Face/GitHub fallback in executable scripts.
