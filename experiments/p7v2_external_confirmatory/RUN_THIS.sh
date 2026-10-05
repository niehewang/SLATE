#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python
ASSET=/data/jx-vmlab/sharegpt_train.json
SHA=fd0a9d8659486e8c1b6ec732368e4a39132adb93fb138bc70fabef834cad2076
URL=https://hf-mirror.com/datasets/heegyu/ShareGPT_Vicuna_unfiltered_no_imsorry/resolve/main/train.json
if [[ ! -f "$ASSET" ]] || [[ "$(sha256sum "$ASSET" | awk '{print $1}')" != "$SHA" ]]; then
  rm -f "$ASSET.part"
  echo '[asset] downloading ShareGPT JSONL from HF-Mirror only...'
  curl -L --fail --retry 8 --retry-delay 3 -C - -o "$ASSET.part" "$URL" || { echo MIRROR_DOWNLOAD_FAILED_BEFORE_SCIENCE; exit 44; }
  GOT=$(sha256sum "$ASSET.part" | awk '{print $1}')
  [[ "$GOT" == "$SHA" ]] || { echo "SHA_MISMATCH got=$GOT expected=$SHA"; rm -f "$ASSET.part"; exit 45; }
  mv "$ASSET.part" "$ASSET"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$ROOT/scripts/run_stagep7.py" 2>&1 | tee -a "$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep7v2_revised_sharegpt_phi_external_confirmatory_v1/launcher.log"
RC=${PIPESTATUS[0]}
RET="$HOME/SLATE_TDSC_SERVER_SHARED/runs/slate_stagep7v2_revised_sharegpt_phi_external_confirmatory_v1/return"
echo "$RC" > "$RET/launcher_exit_code.txt"
tar -czf /data/jx-vmlab/SLATE_SERVER_RETURN_STAGEP7V2_REVISED_SHAREGPT_PHI_EXTERNAL_CONFIRMATORY_V1.tar.gz -C "$RET" .
echo RETURN_PACKAGE=/data/jx-vmlab/SLATE_SERVER_RETURN_STAGEP7V2_REVISED_SHAREGPT_PHI_EXTERNAL_CONFIRMATORY_V1.tar.gz
echo EXIT_CODE=$RC
exit $RC
