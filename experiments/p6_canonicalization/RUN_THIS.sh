#!/usr/bin/env bash
set -uo pipefail
PY=${SLATE_PYTHON:-python}
OUT=${SLATE_WORK_ROOT:-$PWD/.slate_work}/SLATE_SERVER_RETURN_STAGEP6_LLMDNA_CANONICALIZATION_DIAGNOSTIC_V1.tar.gz
RUN="$HOME/SLATE_SHARED/runs/slate_stagep6_llmdna_canonicalization_diagnostic_v1"
mkdir -p "$RUN/return"
set +e
"$PY" scripts/run_stagep6.py 2>&1 | tee "$RUN/run.log"
RC=${PIPESTATUS[0]}
set -e
cp "$RUN/run.log" "$RUN/return/run.log" 2>/dev/null || true
printf '%s\n' "$RC" > "$RUN/return/launcher_exit_code.txt"
tar -czf "$OUT" -C "$RUN/return" .
echo "RETURN_PACKAGE=$OUT"
exit "$RC"
