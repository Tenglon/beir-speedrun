#!/bin/bash
#SBATCH --job-name=beir-enc
#SBATCH --output=/gpfs/scratch/ehpc821/uoa994647/beir_speedrun/logs/enc_%A_%a.out
#SBATCH --error=/gpfs/scratch/ehpc821/uoa994647/beir_speedrun/logs/enc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
set -euo pipefail
source /gpfs/scratch/ehpc821/uoa994647/beir_speedrun/code/bsc/env.sh

eval "$("$VENV/bin/python" -m beirspeed.shard --manifest "$EMB_ROOT/manifest.json" \
  --task "$SLURM_ARRAY_TASK_ID" --print-env)"
echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS shard=$SHARD range=[$START,$END)"

"$VENV/bin/python" -m beirspeed.encode \
  --data-root "$DATA_ROOT" --dataset "$DS" --start "$START" --end "$END" --shard "$SHARD" \
  --out-root "$EMB_ROOT" --model-dir "$MODEL_DIR" \
  --batch-size "${BATCH_SIZE:-256}" --max-length "${MAX_LENGTH:-256}"
