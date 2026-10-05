#!/usr/bin/env bash
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
SHARED="${HOME}/SLATE_TDSC_SERVER_SHARED"
VENDOR="${SHARED}/runs/slate_stagep0_published_baselines_reproduction_v1_1/vendor"
NANOGCG_SRC="${SHARED}/public_baselines_sources/nanoGCG"
export PYTHONPATH="${VENDOR}:${NANOGCG_SRC}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PY" "$PKG_DIR/scripts/selftest.py" || exit 90
"$PY" "$PKG_DIR/scripts/run_analysis_only.py"
RC=$?
"$PY" "$PKG_DIR/scripts/make_return.py" || true
exit "$RC"
