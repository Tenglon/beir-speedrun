# beir_speedrun

Run the full public [BEIR](https://github.com/beir-cellar/beir) benchmark (15
datasets, 33.9M documents) in **~5 minutes of wall-clock time** on a Slurm
cluster with ~100 single-GPU jobs. Built for and validated on BSC MareNostrum5
(H100, fully offline), but the site specifics live in one file (`bsc/env.sh`)
plus the `#SBATCH` headers.

**How it goes fast:** every corpus is split into equal-size shards (500k docs);
one Slurm job array (`%100` throttle) encodes all 88 shards in parallel, then a
second array runs one single-GPU task per dataset doing exact cosine top-100
retrieval + `pytrec_eval` scoring. Everything is idempotent (`.done` markers,
existing `metrics.json` skipped), so re-submitting an array only redoes missing
work.

Measured with `bert-base-uncased` (fp16, cuDNN 9 SDPA / FA3-class attention,
torch 2.11.0+cu126): **~9,800 docs/s per H100**, 88 encode jobs finish in ~3
min wall-clock, retrieval+eval in ~1 min — about 2 GPU-hours total.

## Zero-shot bert-base-uncased baseline (2026-07-19)

Mean pooling over the last hidden layer, cosine similarity, `title + " " + text`
truncated to 256 wordpieces, self-matches dropped, `msmarco` on dev / rest on
test, cqadupstack = average of its 12 sub-forums.

| dataset | nDCG@10 | R@100 | | dataset | nDCG@10 | R@100 |
|---|---:|---:|---|---|---:|---:|
| msmarco (dev) | 0.019 | 0.105 | | quora | 0.610 | 0.861 |
| trec-covid | 0.150 | 0.017 | | dbpedia-entity | 0.041 | 0.105 |
| nfcorpus | 0.049 | 0.106 | | scidocs | 0.029 | 0.143 |
| nq | 0.026 | 0.168 | | fever | 0.036 | 0.157 |
| hotpotqa | 0.083 | 0.223 | | climate-fever | 0.064 | 0.297 |
| fiqa | 0.022 | 0.116 | | scifact | 0.166 | 0.509 |
| arguana | 0.288 | 0.785 | | webis-touche2020 | 0.009 | 0.038 |
| cqadupstack | 0.055 | 0.190 | | **avg (14, excl msmarco)** | **0.116** | **0.265** |

Untrained BERT is a known near-floor baseline — only surface-similarity tasks
(quora, arguana) score meaningfully. Full numbers (nDCG@100, MRR@10, per
sub-forum) in [`results/`](results/).

## Repo layout

```
beirspeed/            python package (stdlib + torch/transformers/numpy/pytrec_eval)
  data.py             BEIR jsonl/tsv loaders
  shard.py            equal-size shard manifest over all corpora
  encode.py           encode one shard -> fp16 .npy (+ids); length-sorted batching
  retrieve.py         per-dataset streaming exact top-k + metrics
  metrics.py          pytrec_eval wrapper + trec_eval-compatible fallback
  aggregate.py        final table (cqadupstack averaging, BEIR ordering)
bsc/                  cluster scripts
  env.sh              ALL site paths (override BEIR_ROOT/BASE_PY/MODEL_DIR)
  setup_env.sh        offline venv + unzip + manifest        (login node, once)
  submit_encode.sh    sbatch array over all shards, %100 throttle
  submit_retrieve.sh  sbatch array, one task per dataset
  status.sh           queue + progress overview
local/
  download_assets.sh  run WITH internet: datasets + model + cp312 wheelhouse
results/              the baseline table above (md + json)
```

## Usage

```bash
# 1. on a machine with internet
bash local/download_assets.sh                  # ~15 GB into ./stage
rsync -a stage/datasets/     user@cluster:$BEIR_ROOT/staging/datasets/
rsync -a stage/models/       user@cluster:$BEIR_ROOT/models/
rsync -a stage/wheelhouse312/ user@cluster:$BEIR_ROOT/staging/wheelhouse312/
rsync -a . user@cluster:$BEIR_ROOT/code/       # this repo

# 2. on the cluster login node
cd $BEIR_ROOT/code
bash bsc/setup_env.sh                          # venv + unzip + shard manifest
bash bsc/submit_encode.sh 85 && bash bsc/submit_retrieve.sh 23   # scifact smoke
bash bsc/submit_encode.sh                      # all 88 shards, <=100 concurrent
bash bsc/submit_retrieve.sh                    # after encode finishes (status.sh)
$VENV/bin/python -m beirspeed.aggregate --results-root $RESULTS_ROOT \
    --out $RESULTS_ROOT/results_table.md
```

To evaluate a different model: drop its HF snapshot under `$BEIR_ROOT/models/`,
`export MODEL_DIR=... EMB_ROOT=... RESULTS_ROOT=...`, resubmit both arrays.

## Porting / offline notes

- Edit `bsc/env.sh` plus the `#SBATCH` partition/account/qos headers in the two
  `.sbatch` files; nothing else is site-specific.
- Driver 535 (CUDA 12.2) ⇒ only cu12x torch builds run (cu13x needs ≥580);
  cu126 works via CUDA minor-version compatibility, and its cuDNN 9.10 provides
  the fused FA3-class SDPA kernels on Hopper (`attn_implementation=sdpa`).
- `local/download_assets.sh` documents the offline-wheel pitfalls (NVIDIA wheels
  moved off PyPI; platform-marker evaluation on the download host; pip's
  `--implementation cp` silently dropping `py3-none-manylinux` wheels).
