#!/bin/bash
#SBATCH --job-name=cb-enc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Submit via submit_colbert_eval.sh — needs CB_MODEL_DIR and CB_EMB_ROOT in env.
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"

eval "$("$VENV/bin/python" -m beirspeed.shard --manifest "$EMB_ROOT/manifest.json" \
  --task "$SLURM_ARRAY_TASK_ID" --print-env)"
echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS shard=$SHARD range=[$START,$END)"

"$VENV/bin/python" -m beirspeed.encode_colbert \
  --data-root "$DATA_ROOT" --dataset "$DS" --start "$START" --end "$END" --shard "$SHARD" \
  --out-root "$CB_EMB_ROOT" --model-dir "$CB_MODEL_DIR" \
  --batch-size "${CB_BATCH_SIZE:-128}" --doc-length "${DOC_LENGTH:-300}"
