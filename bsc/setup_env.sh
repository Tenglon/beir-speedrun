#!/bin/bash
# One-time setup on the login node: python 3.12 venv (miniforge, absolute path so no
# module system needed) with torch 2.11.0+cu126 installed offline from the staged
# wheelhouse — cuDNN 9 SDPA gives FA3-class attention on H100 (driver 535 => cu12x
# builds only, never cu13x). Then unzip datasets and build the shard manifest.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ ! -x "$VENV/bin/python" ]; then
  "$BASE_PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install --no-index --no-cache-dir --find-links "$BEIR_ROOT/staging/wheelhouse312" \
  "torch==2.11.0+cu126" transformers pytrec-eval-terrier numpy
"$VENV/bin/python" - <<'PY'
import numpy, pytrec_eval, torch, transformers
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version(),
      "| transformers", transformers.__version__,
      "| numpy", numpy.__version__, "| pytrec_eval ok")
PY

mkdir -p "$DATA_ROOT"
for z in "$BEIR_ROOT"/staging/datasets/*.zip; do
  name=$(basename "$z" .zip)
  if [ ! -e "$DATA_ROOT/$name" ]; then
    echo "unzipping $name"
    unzip -q -n "$z" -d "$DATA_ROOT"
  fi
done

"$VENV/bin/python" -m beirspeed.shard --data-root "$DATA_ROOT" \
  --out "$EMB_ROOT/manifest.json" --shard-size "${SHARD_SIZE:-500000}"
echo "setup complete"
