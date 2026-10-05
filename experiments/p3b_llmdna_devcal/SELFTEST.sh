#!/usr/bin/env bash
set -euo pipefail
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
ROOT="$(cd "$(dirname "$0")" && pwd)"
$PY -m py_compile "$ROOT/scripts/run_stagep3b.py"
$PY "$ROOT/scripts/run_stagep3b.py" --selftest
$PY - <<'PY2' "$ROOT/config/project.json"
import json,sys
c=json.load(open(sys.argv[1]))
assert c['candidate_dev']==512
assert c['heldout_access'] is False
assert c['main_q']==64
print('CONFIG_GUARD_OK')
PY2
if grep -R -nE 'git clone|git fetch|https://github.com|huggingface\.co' "$ROOT/RUN_THIS.sh" "$ROOT/scripts"; then
  echo 'FORBIDDEN_NETWORK_REFERENCE_IN_EXECUTABLE' >&2; exit 2
fi
if ! grep -q "P3B held-out hard guard" "$ROOT/scripts/run_stagep3b.py"; then
  echo 'HELDOUT_GUARD_MISSING' >&2; exit 3
fi
echo SELFTEST_OK
