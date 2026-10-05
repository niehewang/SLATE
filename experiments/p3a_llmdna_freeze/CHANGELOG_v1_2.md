# StageP3A v1.2 — Official Release Alignment

This is the final recovery of the same P3A freeze stage, not a new preflight stage.

## Why
The v1.1 run proved that PyPI `llm-dna==0.2.3` installs correctly but does not expose `TextDNAExtractor`, while the corrected current public LLM-DNA pipeline used for the planned Full-SLATE adapter is the release exposing `TextDNAExtractor` with Qwen3-Embedding-8B and shared Gaussian projection. P3A therefore aligns the freeze to official PyPI `llm-dna==1.0.1`.

## Network policy
- No GitHub access at runtime.
- Prefer an already available local v1.0.1 tag only if present.
- Otherwise use official PyPI `llm-dna==1.0.1`.
- Reuse the already complete local Qwen3-Embedding-8B.

## Scientific settings unchanged
q=64; 128-D DNA; pre-aggregation 64; shared Gaussian projection seed 42; normalize_embeddings=False; Qwen3-Embedding-8B; source and NF4 runtime views.
