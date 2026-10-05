#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
set +e
"$PY" scripts/run_stagep2c.py
RC=$?
"$PY" scripts/make_return.py
exit $RC
