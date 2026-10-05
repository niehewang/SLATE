#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep0_published_baselines_reproduction_v1_1/vendor:${PYTHONPATH:-}"
mkdir -p "$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2b_llmprint_selective_serving_degradation_v1/return"
set +e
"$PY" "$ROOT/scripts/run_stagep2b.py" 2>&1 | tee -a "$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2b_llmprint_selective_serving_degradation_v1/run.log"
RC=${PIPESTATUS[0]}
"$PY" "$ROOT/scripts/make_return.py"
exit $RC
