#!/bin/bash
# Submit the encode job array over all manifest shards, throttled to <=100 concurrent.
# Usage: submit_encode.sh [task_spec]   e.g. submit_encode.sh 0-87  (default: all)
set -euo pipefail
source "$(dirname "$0")/env.sh"

N=$("$VENV/bin/python" -c "import json;print(len(json.load(open('$EMB_ROOT/manifest.json'))['tasks']))")
SPEC="${1:-0-$((N - 1))}"
echo "manifest has $N shard tasks; submitting array=$SPEC%${THROTTLE:-100}"
sbatch --array="$SPEC%${THROTTLE:-100}" ${SBATCH_EXTRA:-} "$CODE_ROOT/bsc/encode_array.sbatch"
