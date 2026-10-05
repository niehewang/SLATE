# SLATE StageP3A v1.2 — LLM-DNA Official Freeze Official-Release Alignment

Purpose: complete the same one-time P3A official-coordinate freeze after v1 failed before any scientific computation because the server cannot access GitHub.

Server network rule: **this package never contacts GitHub**. It first exports tag `v1.0.1` from the StageP0 local LLM-DNA clone. If that tag is not available locally, it downloads the official `llm-dna==1.0.1` wheel from PyPI. The only potentially large network transfer is Qwen3-Embedding-8B through `hf-mirror` if no complete local copy exists.

Scientific settings are unchanged from P3A v1. On success, go directly to StageP3B Full-SLATE Dev/Cal Freeze. Do not create another P3A/preflight stage.
