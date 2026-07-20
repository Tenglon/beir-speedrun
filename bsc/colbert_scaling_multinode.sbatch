#!/bin/bash
#SBATCH --job-name=cb-scale-mn
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=00:40:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Multi-node strong scaling at FIXED global batch 32.
# Submit with --nodes=N; micro-batch shrinks to keep 32 global:
#   2 nodes (8 GPU) -> bs 4      4 nodes (16 GPU) -> bs 2
# env in: BS (micro-batch), STEPS (default 160).
set -uo pipefail
source "$CODE_ROOT/bsc/env.sh"
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$((SLURM_JOB_NUM_NODES * 4))
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=29511
echo "=== scaling nodes=$SLURM_JOB_NUM_NODES gpus=$NGPU micro_bs=${BS:-4} global=$(( ${BS:-4} * NGPU )) master=$MASTER ==="

rm -rf "$BEIR_ROOT/colbert/scale_${NGPU}gpu"
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --gres=gpu:4 \
  "$VENV/bin/torchrun" \
    --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER:29511" \
    "$CODE_ROOT/train/train_colbert.py" \
      --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
      --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
      --out-dir "$BEIR_ROOT/colbert/scale_${NGPU}gpu" \
      --batch-size "${BS:-4}" --max-steps "${STEPS:-160}" --epochs 1 \
  || echo "gpus=$NGPU FAILED rc=$?"
