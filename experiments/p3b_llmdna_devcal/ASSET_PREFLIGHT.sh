#!/usr/bin/env bash
set -euo pipefail
PY=/path/to/shared_runtime/envs/fas-py311/bin/python
$PY -c "from pathlib import Path; candidates=[Path('/path/to/slate_shared/public_models/Phi-3-mini-4k-instruct'),Path('/path/to/slate_shared/public_models/Phi-3-mini-4k-instruct')]; complete=lambda p:(p/'config.json').exists() and bool(list(p.glob('*.safetensors'))+list(p.glob('*.bin'))); print('PHI_ASSET_CANDIDATES'); [print(('OK  ' if complete(p) else 'MISS'),p) for p in candidates]; sel=next((p for p in candidates if complete(p)),None); print('SELECTED_PHI=',sel); assert sel is not None, 'NO_COMPLETE_PHI_CHECKPOINT'; print('ASSET_PREFLIGHT_OK')"
