# beir_speedrun

Fast BEIR benchmark evaluation on BSC MareNostrum5 (offline cluster, Slurm, up to
~100 concurrent single-GPU jobs). Every dataset's corpus is split into equal-size
shards (default 500k docs); one job array encodes all shards in parallel, then a
second array does per-dataset exact cosine top-k retrieval + trec_eval metrics.

## Zero-shot baseline configuration

- Model: `bert-base-uncased`, no fine-tuning (zero-shot)
- Embedding: mean pooling over last hidden layer (attention-mask weighted), fp16 storage
- Text: `title + " " + text`, truncated to 256 wordpieces (queries likewise)
- Similarity: cosine; exact search, top-100; self-matches (doc id == query id) dropped
- Datasets: 15 public BEIR datasets (cqadupstack = 12 sub-forums averaged);
  `msmarco` scored on dev, everything else on test
- Metrics: nDCG@10 (primary), nDCG@100, Recall@100, MRR@10 via `pytrec_eval`

## Layout on BSC

Everything lives under `/gpfs/scratch/ehpc821/uoa994647/beir_speedrun`:
`code/` (this repo), `staging/` (uploaded zips, model, wheelhouse), `datasets/`,
`models/`, `venv/`, `embeddings/` (shards + manifest), `results/`, `logs/`.

The cluster has **no internet on any node** (login, compute, transfer), so all
assets are downloaded locally and rsync'd to `transfer1.bsc.es`. The venv is
python 3.12 (miniforge) with **torch 2.11.0+cu126** installed offline from
`staging/wheelhouse312` — cuDNN 9 SDPA provides FA3-class fused attention on
H100. GPU driver is 535 (CUDA 12.2), so only cu12x torch builds work (cu13x
needs driver >= 580); cu126 runs via CUDA minor-version compatibility.

## Run order (login node)

```bash
cd /gpfs/scratch/ehpc821/uoa994647/beir_speedrun/code
bash bsc/setup_env.sh          # venv + unzip + shard manifest
bash bsc/submit_encode.sh 0    # smoke: one scifact-sized shard
bash bsc/submit_retrieve.sh    # after encode; or a single-task smoke first
bash bsc/submit_encode.sh      # full array, %100 throttle
bash bsc/status.sh
$VENV/bin/python -m beirspeed.aggregate --results-root $RESULTS_ROOT --out $RESULTS_ROOT/results_table.md
```

Jobs use `--partition=acc --account=ehpc821 --qos=acc_ehpc --gres=gpu:1
--cpus-per-task=20` (H100). Everything is idempotent via `.done` markers /
existing `metrics.json`: rerunning an array only redoes missing work.
