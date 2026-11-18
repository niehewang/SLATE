#!/usr/bin/env bash
set -uo pipefail
PY=${SLATE_PYTHON:-python}
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$HOME/SLATE_SHARED/runs/slate_stagep4b_srp_fullslate_heldout_confirmatory_v1_1_engineering_recovery"
RET="$RUN/return"
OUT=${SLATE_WORK_ROOT:-$PWD/.slate_work}/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP4B_SRP_FULLSLATE_HELDOUT_CONFIRMATORY_V1_1_ENGINEERING_RECOVERY.tar.gz
mkdir -p "$RET"
cd "$ROOT"
"$PY" scripts/run_stagep4b.py 2>&1 | tee -a "$RUN/run.log"
RC=${PIPESTATUS[0]}
echo "$RC" > "$RET/launcher_exit_code.txt"
cp -f "$RUN/run.log" "$RET/run.log" 2>/dev/null || true
tar -czf "$OUT" -C "$RET" .
echo "RETURN_PACKAGE=$OUT"
exit "$RC"
