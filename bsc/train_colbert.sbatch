#!/bin/bash
#SBATCH --job-name=colbert-train
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Submit with --nodes=N (default 1). env in: OUT_DIR (required);
# BS/LR/EPOCHS/MAX_STEPS/SAVE_STEPS optional.
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"
export HF_DATASETS_OFFLINE=1
# DDP replicas need ~1GB headroom over the single-GPU footprint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NNODES=${SLURM_JOB_NUM_NODES:-1}
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "nodes=$NNODES gpus=$((NNODES * 4)) micro_bs=${BS:-8} global=$(( ${BS:-8} * NNODES * 4 ))"

srun --ntasks="$NNODES" --ntasks-per-node=1 --gres=gpu:4 \
  "$VENV/bin/torchrun" \
    --nnodes="$NNODES" --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER:29521" \
    "$CODE_ROOT/train/${TRAIN_SCRIPT:-train_colbert.py}" \
      --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
      --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
      --out-dir "$OUT_DIR" \
      --batch-size "${BS:-8}" --lr "${LR:-1e-5}" \
      --epochs "${EPOCHS:-3}" --max-steps "${MAX_STEPS:--1}" \
      --save-steps "${SAVE_STEPS:-0}"
