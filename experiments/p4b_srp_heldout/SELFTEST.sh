#!/usr/bin/env bash
set -euo pipefail
PY=/path/to/shared_runtime/envs/fas-py311/bin/python
"$PY" -m py_compile scripts/p4a_core.py scripts/run_stagep4b.py
"$PY" scripts/run_stagep4b.py --selftest
