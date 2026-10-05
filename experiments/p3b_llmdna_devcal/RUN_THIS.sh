#!/usr/bin/env bash
set -uo pipefail
export SLATE_PYTHON=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1"
RET="$RUN/return"
mkdir -p "$RET"

set +e
"$SLATE_PYTHON" "$ROOT/scripts/run_stagep3b.py" 2>&1 | tee -a "$RUN/launcher.log"
RC=${PIPESTATUS[0]}
set -e
cp -f "$RUN/run.log" "$RET/run.log" 2>/dev/null || true
printf '%s\n' "$RC" > "$RET/launcher_exit_code.txt"
OUT=/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP3B_LLMDNA_FULLSLATE_DEVCAL_FREEZE_V1_2_1.tar.gz
tar -czf "$OUT" -C "$RET" .
echo "RETURN=$OUT"
echo "PYTHON_EXIT_CODE=$RC"
exit "$RC"
