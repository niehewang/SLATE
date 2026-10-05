#!/usr/bin/env bash
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SLATE_PYTHON:-/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python}"
RUN_ROOT="${SLATE_P3A_RUN:-$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3a_llmdna_official_freeze_v1_2}"
RET="/data/jx-vmlab/SLATE_SERVER_RETURN_PUBLISHED_BASELINES_STAGEP3A_LLMDNA_OFFICIAL_FREEZE_V1_2.tar.gz"
mkdir -p "$RUN_ROOT"
export SLATE_P3A_RUN="$RUN_ROOT"
export SLATE_PYTHON="$PYTHON"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Explicit network policy: never contact GitHub from this workflow.
export GIT_TERMINAL_PROMPT=0
set +e
"$PYTHON" "$PKG_DIR/scripts/run_stagep3a.py" 2>&1 | tee "$RUN_ROOT/run.log"
RC=${PIPESTATUS[0]}
set -e
cd "$RUN_ROOT"
FILES=(status.json environment.json official_release_freeze.json embedding_asset.json api_introspection.json source_q64.json nf4_q64.json official_distance_q64.json adapter_preflight.json p3b_freeze_manifest.json paper_result_summary.json summary.json run.log source_run.json nf4_run.json adapter_preflight_run.json)
KEEP=()
for f in "${FILES[@]}"; do [[ -f "$f" ]] && KEEP+=("$f"); done
if [[ ${#KEEP[@]} -eq 0 ]]; then echo "No return artifacts produced" >&2; exit 97; fi
tar -czf "$RET" "${KEEP[@]}"
echo "RETURN_FILE=$RET"
echo "STAGE_RC=$RC"
exit "$RC"
