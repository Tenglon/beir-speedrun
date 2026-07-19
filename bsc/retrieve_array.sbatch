#!/bin/bash
#SBATCH --job-name=beir-ret
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Submit via submit_retrieve.sh — the job inherits BEIR_ROOT/CODE_ROOT/... from it.
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EMB_ROOT/retrieve_targets.tsv")
DS=$(echo "$line" | cut -f1)
SPLIT=$(echo "$line" | cut -f2)
NSHARDS=$(echo "$line" | cut -f3)
echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS split=$SPLIT shards=$NSHARDS"

"$VENV/bin/python" -m beirspeed.retrieve \
  --data-root "$DATA_ROOT" --dataset "$DS" --split "$SPLIT" \
  --emb-root "$EMB_ROOT" --out-root "$RESULTS_ROOT" --model-dir "$MODEL_DIR" \
  --k 100 --expected-shards "$NSHARDS"
