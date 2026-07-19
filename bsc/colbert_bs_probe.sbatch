#!/bin/bash
#SBATCH --job-name=colbert-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:30:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# Single-GPU batch-size sweep (powers of two, bf16 + SDPA): 20 KD steps per size,
# report peak memory + step throughput so the 4-GPU run can use the largest fit.
set -uo pipefail
source "$CODE_ROOT/bsc/env.sh"
export HF_DATASETS_OFFLINE=1

for bs in 16 32 64 128 256; do
  echo "=== probe bs=$bs ==="
  rm -rf "$BEIR_ROOT/colbert/probe_bs$bs"
  start=$(date +%s)
  "$VENV/bin/python" "$CODE_ROOT/train/train_colbert.py" \
    --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
    --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
    --out-dir "$BEIR_ROOT/colbert/probe_bs$bs" \
    --batch-size "$bs" --max-steps 20 --epochs 1 \
    && echo "bs=$bs OK elapsed=$(( $(date +%s) - start ))s" \
    || echo "bs=$bs FAILED (likely OOM)"
done
