#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${SLATE_PYTHON:-python}"
"$PY" "$ROOT/scripts/selftest.py"
exec "$PY" "$ROOT/scripts/run_operating_region.py" "$@"
