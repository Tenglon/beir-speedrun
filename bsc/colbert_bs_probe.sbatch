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

for cfg in "16 4 --no-math-sdp" "32 4 --no-math-sdp" "64 4 --no-math-sdp" "128 8 --no-math-sdp" "16 4 "; do
  read -r bs nw flag <<< "$cfg"
  echo "=== probe bs=$bs nw=$nw flag=${flag:-none} ==="
  rm -rf "$BEIR_ROOT/colbert/probe_bs${bs}_nw${nw}${flag:+_nomath}"
  start=$(date +%s)
  timeout 900 "$VENV/bin/python" "$CODE_ROOT/train/train_colbert.py" \
    --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
    --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
    --out-dir "$BEIR_ROOT/colbert/probe_bs${bs}_nw${nw}${flag:+_nomath}" \
    --batch-size "$bs" --max-steps 20 --epochs 1 --num-workers "$nw" $flag \
    && echo "bs=$bs nw=$nw flag=${flag:-none} OK elapsed=$(( $(date +%s) - start ))s" \
    || echo "bs=$bs nw=$nw flag=${flag:-none} FAILED rc=$? (OOM or 900s timeout)"
done
