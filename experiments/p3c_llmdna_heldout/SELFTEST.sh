#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/path/to/shared_runtime/envs/fas-py311/bin/python
"$PY" -m py_compile "$ROOT/scripts/p3b_core.py" "$ROOT/scripts/run_stagep3c.py"
"$PY" "$ROOT/scripts/run_stagep3c.py" --selftest
