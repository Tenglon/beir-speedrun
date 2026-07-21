#!/bin/bash
#SBATCH --job-name=halo-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:30:00
#SBATCH --partition=acc
#SBATCH --account=ehpc821
#SBATCH --qos=acc_ehpc
# End-to-end HALO smoke on one GPU: short LoRA+lift KD train, checkpoint reload,
# scifact encode + Lorentz retrieval. Validates: peft merge/load, curvature learning,
# lifted-embedding storage, negative-Lorentz-distance MaxSim, metrics sanity.
set -euo pipefail
source "$CODE_ROOT/bsc/env.sh"
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SMOKE=$BEIR_ROOT/colbert/halo_smoke
rm -rf "$SMOKE"
echo "=== train 300 steps ==="
"$VENV/bin/python" "$CODE_ROOT/train/train_colbert_halo.py" \
  --model-dir "$BEIR_ROOT/models/bert-base-uncased" \
  --data-dir "$BEIR_ROOT/kd_dataset/ms-marco-en-bge-gemma" \
  --out-dir "$SMOKE" --batch-size 8 --max-steps 300 --epochs 1

echo "=== reload + encode scifact ==="
"$VENV/bin/python" -m beirspeed.encode_colbert \
  --data-root "$DATA_ROOT" --dataset scifact --start 0 --end 5183 --shard 0 \
  --out-root "$SMOKE/emb" --model-dir "$SMOKE/final" --batch-size 128

echo "=== lorentz retrieval scifact ==="
"$VENV/bin/python" -m beirspeed.retrieve_colbert \
  --data-root "$DATA_ROOT" --dataset scifact --split test \
  --emb-root "$SMOKE/emb" --out-root "$SMOKE/results" --model-dir "$SMOKE/final" \
  --k 100 --expected-shards 1
echo "=== smoke done ==="
