#!/bin/bash
# Convert one HALO trainer checkpoint (adapter form) to a merged model, then
# submit its small6 eval. Idempotent: skips conversion if merged model exists.
# Usage: convert_and_eval_halo.sh <step> [run_dir] [tag_prefix]
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEP=$1
RUN=${2:-$BEIR_ROOT/colbert/run3_halo}
PFX=${3:-r3s}

CK=$RUN/checkpoint-$STEP
OUT=$RUN/merged-$STEP
if [ ! -f "$OUT/modules.json" ]; then
  "$VENV/bin/python" "$CODE_ROOT/train/convert_halo_checkpoint.py" \
    --base "$BEIR_ROOT/models/bert-base-uncased" --checkpoint "$CK" --out "$OUT"
fi
bash "$CODE_ROOT/bsc/submit_colbert_eval.sh" "$PFX$STEP" "$OUT" small6
