# Site configuration — the single place with cluster-specific paths.
# Override BEIR_ROOT / BASE_PY / MODEL_DIR via environment to port or swap models.
export BEIR_ROOT=${BEIR_ROOT:-/gpfs/scratch/ehpc821/uoa994647/beir_speedrun}
export BASE_PY=${BASE_PY:-/apps/ACC/MINIFORGE/25.3.0-3/bin/python}

export CODE_ROOT=$BEIR_ROOT/code
export DATA_ROOT=$BEIR_ROOT/datasets
export EMB_ROOT=$BEIR_ROOT/embeddings
export RESULTS_ROOT=$BEIR_ROOT/results
export MODEL_DIR=${MODEL_DIR:-$BEIR_ROOT/models/bert-base-uncased}
export VENV=$BEIR_ROOT/venv
export LOGS=$BEIR_ROOT/logs

# no internet on any BSC node; also keep every cache off the (nearly full) $HOME
export HF_HOME=$BEIR_ROOT/cache/huggingface
export TMPDIR=$BEIR_ROOT/cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$EMB_ROOT" "$RESULTS_ROOT" "$LOGS" "$HF_HOME" "$TMPDIR"
