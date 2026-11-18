#!/usr/bin/env bash
set -euo pipefail
PY=/path/to/shared_runtime/envs/fas-py311/bin/python
"$PY" -m py_compile scripts/run_stagep4a.py
"$PY" scripts/run_stagep4a.py --selftest
