#!/usr/bin/env bash
set -euo pipefail
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
"$PY" -m py_compile scripts/run_stagep4a.py
"$PY" scripts/run_stagep4a.py --selftest
