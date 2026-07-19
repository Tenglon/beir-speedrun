#!/bin/bash
#SBATCH --job-name=beir-enc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Submit via submit_encode.sh — the job inherits BEIR_ROOT/CODE_ROOT/... from it.
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"

eval "$("$VENV/bin/python" -m beirspeed.shard --manifest "$EMB_ROOT/manifest.json" \
  --task "$SLURM_ARRAY_TASK_ID" --print-env)"
echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS shard=$SHARD range=[$START,$END)"

"$VENV/bin/python" -m beirspeed.encode \
  --data-root "$DATA_ROOT" --dataset "$DS" --start "$START" --end "$END" --shard "$SHARD" \
  --out-root "$EMB_ROOT" --model-dir "$MODEL_DIR" \
  --batch-size "${BATCH_SIZE:-256}" --max-length "${MAX_LENGTH:-256}"
