#!/bin/bash
#SBATCH --job-name=cone-enc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:30:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# env in: CONE_CB_DIR, CONE_PT, CONE_EMB (+ inherited BEIR_ROOT/CODE_ROOT/...)
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"

eval "$("$VENV/bin/python" -m beirspeed.shard --manifest "$EMB_ROOT/manifest.json" \
  --task "$SLURM_ARRAY_TASK_ID" --print-env)"
echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS shard=$SHARD"

"$VENV/bin/python" -m beirspeed.encode_cone \
  --data-root "$DATA_ROOT" --dataset "$DS" --start "$START" --end "$END" --shard "$SHARD" \
  --out-root "$CONE_EMB" --colbert-dir "$CONE_CB_DIR" --cone-pt "$CONE_PT"
