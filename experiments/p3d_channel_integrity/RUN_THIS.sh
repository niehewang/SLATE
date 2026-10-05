#!/usr/bin/env bash
set -uo pipefail
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
[[ -x "$PY" ]] || PY=python
BASE=/data/jx-vmlab
RET="$BASE/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP3D_LLMDNA_CHANNEL_INTEGRITY_V1.tar.gz"
RUNROOT="$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3d_llmdna_channel_integrity_v1"
mkdir -p "$RUNROOT/return"
"$PY" scripts/run_stagep3d.py 2>&1 | tee "$RUNROOT/launcher.log"
RC=${PIPESTATUS[0]}
echo "$RC" > "$RUNROOT/return/launcher_exit_code.txt"
cp -f "$RUNROOT/run.log" "$RUNROOT/return/run.log" 2>/dev/null || true
tar -czf "$RET" -C "$RUNROOT/return" .
echo "RETURN_PACKAGE=$RET"
exit "$RC"
