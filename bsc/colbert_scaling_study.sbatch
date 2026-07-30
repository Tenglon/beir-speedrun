#!/bin/bash
#SBATCH --job-name=colbert-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=01:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Strong-scaling study at FIXED global batch 32 (micro-batch 8 everywhere):
#   1 GPU x accum 4  |  2 GPU x accum 2  |  4 GPU x accum 1
# Same node, same GPUs, sequential; s/optimizer-step is directly comparable.
set -uo pipefail
source "$CODE_ROOT/bsc/env.sh"
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  local ngpu=$1 accum=$2 steps=$3 devs=$4 port=$5
  echo "=== scaling ngpu=$ngpu accum=$accum global_batch=32 ==="
  rm -rf "$BEIR_ROOT/colbert/scale_${ngpu}gpu"
  CUDA_VISIBLE_DEVICES=$devs timeout 1200 "$VENV/bin/torchrun" \
    --nproc_per_node="$ngpu" --master_port="$port" "$CODE_ROOT/train/train_colbert.py" \
    --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
    --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
    --out-dir "$BEIR_ROOT/colbert/scale_${ngpu}gpu" \
    --batch-size 8 --grad-accum "$accum" --max-steps "$steps" --epochs 1 \
    || echo "ngpu=$ngpu FAILED rc=$?"
}

run 1 4 40  0       29501
run 2 2 80  0,1     29502
run 4 1 160 0,1,2,3 29503
