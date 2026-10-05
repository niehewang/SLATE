#!/usr/bin/env bash
set -euo pipefail
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
[[ -x "$PY" ]] || PY=python
"$PY" -m py_compile scripts/p3b_core.py scripts/p3c_helpers.py scripts/run_stagep3d.py
"$PY" scripts/run_stagep3d.py --selftest
