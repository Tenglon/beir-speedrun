#!/bin/bash
#SBATCH --job-name=colbert-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# env in: OUT_DIR (required); BS/LR/EPOCHS/MAX_STEPS/NPROC optional.
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"
export HF_DATASETS_OFFLINE=1
# DDP replicas need ~1GB headroom over the single-GPU footprint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$VENV/bin/torchrun" --nproc_per_node="${NPROC:-4}" "$CODE_ROOT/train/train_colbert.py" \
  --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
  --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
  --out-dir "$OUT_DIR" \
  --batch-size "${BS:-32}" --lr "${LR:-1e-5}" \
  --epochs "${EPOCHS:-3}" --max-steps "${MAX_STEPS:--1}"
