#!/usr/bin/env bash
set -euo pipefail
PY=/path/to/shared_runtime/envs/fas-py311/bin/python
[[ -x "$PY" ]] || PY=python
"$PY" -m py_compile scripts/p3b_core.py scripts/p3c_helpers.py scripts/run_stagep3d.py
"$PY" scripts/run_stagep3d.py --selftest
