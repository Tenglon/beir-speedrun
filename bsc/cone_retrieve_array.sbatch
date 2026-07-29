#!/bin/bash
#SBATCH --job-name=cone-ret
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# env in: CONE_CB_DIR, CONE_PT, CONE_EMB, CONE_RES
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONE_EMB/retrieve_targets.tsv")
DS=$(echo "$line" | cut -f1)
SPLIT=$(echo "$line" | cut -f2)
NSHARDS=$(echo "$line" | cut -f3)
echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS split=$SPLIT shards=$NSHARDS"

"$VENV/bin/python" -m beirspeed.retrieve_cone \
  --data-root "$DATA_ROOT" --dataset "$DS" --split "$SPLIT" \
  --emb-root "$CONE_EMB" --out-root "$CONE_RES" \
  --colbert-dir "$CONE_CB_DIR" --cone-pt "$CONE_PT" --expected-shards "$NSHARDS"
