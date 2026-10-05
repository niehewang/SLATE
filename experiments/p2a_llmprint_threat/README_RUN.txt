SLATE StageP2A v1.3 — analysis-only suffix-decode hotfix

Run:
  chmod +x RUN_THIS.sh
  ./RUN_THIS.sh

This package reuses the 128/128 frozen LLMPrint GCG suffix files already present under:
  ~/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2a_llmprint_threat_surface_v1/llmprint_128/

It does NOT rerun token-pair generation, Qwen loading, or GCG.

Expected return:
  /data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP2A_LLMPRINT_THREAT_SURFACE_V1_3.tar.gz
