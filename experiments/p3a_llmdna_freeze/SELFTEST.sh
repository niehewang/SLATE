#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
python -m py_compile "$PKG_DIR/scripts/"*.py
bash -n "$PKG_DIR/RUN_THIS.sh"
if grep -R -nE 'git clone|git fetch|https://github\.com' "$PKG_DIR/scripts" "$PKG_DIR/RUN_THIS.sh"; then
  echo 'FAIL: GitHub network dependency found in executable workflow' >&2
  exit 2
fi
grep -q "llm-dna==1.0.1" "$PKG_DIR/scripts/run_stagep3a.py"
grep -q "git.*archive" "$PKG_DIR/scripts/run_stagep3a.py"
echo SELFTEST_OK
