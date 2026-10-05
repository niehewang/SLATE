#!/usr/bin/env bash
set -uo pipefail
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
OUT=/data/jx-vmlab/SLATE_SERVER_RETURN_STAGEP5_JOINT_FROZEN_ABLATION_CLAIM_BOUNDARY_V1_1_CACHE_TAG_RECOVERY.tar.gz
RUN="$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep5_joint_frozen_ablation_claim_boundary_v1_1_cache_tag_recovery"
mkdir -p "$RUN/return"
set +e
"$PY" scripts/run_stagep5.py 2>&1 | tee "$RUN/run.log"
RC=${PIPESTATUS[0]}
set -e
cp "$RUN/run.log" "$RUN/return/run.log" 2>/dev/null || true
printf '%s\n' "$RC" > "$RUN/return/launcher_exit_code.txt"
tar -czf "$OUT" -C "$RUN/return" .
echo "RETURN_PACKAGE=$OUT"
exit "$RC"
