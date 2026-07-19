#!/bin/bash
# One-time setup on the login node: venv over wote-h100's python (reuses its torch),
# offline install of transformers + pytrec_eval from the staged wheelhouse,
# unzip datasets, build the shard manifest.
set -euo pipefail
source "$(dirname "$0")/env.sh"

BASE_PY=/gpfs/scratch/ehpc821/uoa994647/conda_envs/wote-h100/bin/python

if [ ! -x "$VENV/bin/python" ]; then
  "$BASE_PY" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install --no-index --find-links "$BEIR_ROOT/staging/wheelhouse" \
  transformers pytrec-eval-terrier
"$VENV/bin/python" - <<'PY'
import torch, transformers, pytrec_eval, numpy
print("torch", torch.__version__, "| transformers", transformers.__version__,
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
