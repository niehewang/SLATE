# Protocol — StageP3A v1.2

Same scientific protocol as P3A v1; engineering recovery only. Freeze official LLM-DNA v1.0.1, TextDNAExtractor, Qwen3-Embedding-8B, q=64, 128-D DNA, 64-D pre-aggregation slice, shared Gaussian projection seed 42, normalize_embeddings=False. Run Qwen3-4B-Base source and NF4 runtime descendant using the same ordered probes. Verify the public embeddings-to-DNA adapter. Do not access held-out traffic and do not select lineage/risk/detector thresholds in P3A.

The return additionally contains `p3b_freeze_manifest.json` and `paper_result_summary.json` required by Experiment Direction v2. `p3b_freeze_manifest.json` freezes the P3B candidate grids/budgets and explicitly leaves J_E/lambda choice/native threshold to P3B development/calibration.
