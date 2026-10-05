#!/usr/bin/env bash
set -uo pipefail
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep4b_srp_fullslate_heldout_confirmatory_v1_1_engineering_recovery"
RET="$RUN/return"
OUT=/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP4B_SRP_FULLSLATE_HELDOUT_CONFIRMATORY_V1_1_ENGINEERING_RECOVERY.tar.gz
mkdir -p "$RET"
cd "$ROOT"
"$PY" scripts/run_stagep4b.py 2>&1 | tee -a "$RUN/run.log"
RC=${PIPESTATUS[0]}
echo "$RC" > "$RET/launcher_exit_code.txt"
cp -f "$RUN/run.log" "$RET/run.log" 2>/dev/null || true
tar -czf "$OUT" -C "$RET" .
echo "RETURN_PACKAGE=$OUT"
exit "$RC"
