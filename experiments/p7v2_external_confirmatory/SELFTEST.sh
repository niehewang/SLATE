#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
# hard prohibition on official endpoints in executable shells
if grep -R -nE 'https?://(www\.)?(huggingface\.co|github\.com)' "$ROOT"/*.sh "$ROOT"/scripts/*.py 2>/dev/null; then
  echo 'ERROR: official HF/GitHub endpoint found'; exit 2
fi
"$PY" -m py_compile "$ROOT/scripts/run_stagep7.py" "$ROOT/scripts/p3b_core.py"
"$PY" "$ROOT/scripts/run_stagep7.py" --selftest
echo SELFTEST_OK
