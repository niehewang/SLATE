#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=${SLATE_PYTHON:-python}
RUN=${SLATE_SHARED_ROOT:-$PWD/.slate_shared}/runs/slate_stagep3c_llmdna_fullslate_heldout_confirmatory_v1
RET="$RUN/return"
mkdir -p "$RET"
LOG="$RUN/launcher.log"
set +e
"$PY" "$ROOT/scripts/run_stagep3c.py" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$RC" > "$RET/launcher_exit_code.txt"
cp -f "$RUN/run.log" "$RET/run.log" 2>/dev/null || true
OUT=${SLATE_WORK_ROOT:-$PWD/.slate_work}/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP3C_LLMDNA_FULLSLATE_HELDOUT_CONFIRMATORY_V1.tar.gz
mkdir -p ${SLATE_WORK_ROOT:-$PWD/.slate_work}
if [ -d "$RET" ]; then
  tar -czf "$OUT" -C "$RET" .
  echo "RETURN_PACKAGE=$OUT"
fi
exit "$RC"
