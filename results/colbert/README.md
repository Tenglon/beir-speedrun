# ColBERT (bert-base-uncased) trained on ms-marco-en-bge-gemma

Knowledge distillation with PyLate: teacher scores from **bge-reranker-v2-gemma**,
negatives mined by ColBERT (`lightonai/ms-marco-en-bge-gemma`, 640k listwise
examples, n_ways=32). bf16 + cuDNN SDPA, lr 1e-5, doc length 300, dim 128.

Two runs, 60000 optimizer steps each, evaluated asynchronously during training on
6 small BEIR datasets (scifact / trec-covid / nfcorpus / fiqa / arguana / scidocs):

| steps | run1: 4xH100, global 32 | run2: 8xH100, global 64 |
|---:|---:|---:|
| 10k | — | 0.4524 |
| 20k | 0.4572 (ep1) | 0.4621 |
| 30k | — | 0.4614 |
| 40k | 0.4617 (ep2, run1 peak) | 0.4619 |
| 50k | — | 0.4638 |
| 60k | 0.4608 (ep3) | **0.4640 (best)** |

run1 peaks at epoch 2 then dips (small-batch overfitting); run2's larger batch +
2x data keeps improving through 60k steps. Best checkpoint: `run2/checkpoint-60000`.

## Full BEIR (best checkpoint) vs zero-shot baseline, nDCG@10

| dataset | zero-shot bert-base | ColBERT (this run) |
|---|---:|---:|
| msmarco (dev) | 0.019 | 0.456 |
| trec-covid | 0.150 | 0.717 |
| nfcorpus | 0.049 | 0.352 |
| nq | 0.026 | 0.589 |
| hotpotqa | 0.083 | 0.736 |
| fiqa | 0.022 | 0.378 |
| arguana | 0.288 | 0.441 |
| webis-touche2020 | 0.009 | 0.305 |
| cqadupstack | 0.055 | 0.381 |
| quora | 0.610 | 0.834 |
| dbpedia-entity | 0.041 | 0.470 |
| scidocs | 0.029 | 0.168 |
| fever | 0.036 | 0.808 |
| climate-fever | 0.064 | 0.213 |
| scifact | 0.166 | 0.728 |
| **avg (14, excl msmarco)** | **0.116** | **0.5085** |

Reference points (bert-base backbone, BEIR avg): ColBERTv2 ~0.497 on its 13-dataset
subset; this run reaches 0.5085 on 14 — the stronger KD dataset (LLM-grade teacher,
listwise scores, ColBERT-mined negatives) does the work the plain mined-triplets
collection cannot. Full per-metric numbers in `results_table.md`.

## run1 vs run2 on full BEIR

Both best checkpoints were also compared on the full suite (run1 = checkpoint-40000,
its small6 peak; table in `results_table_run1.md`): avg nDCG@10 **0.5038 (run1)** vs
**0.5085 (run2)**. The small6 gap (+0.0023) held and widened (+0.0047) at full scale
— run2 wins on 12 of 15 datasets (run1 ahead only on fever +0.007, cqadupstack and
scidocs by <0.001), confirming both the checkpoint selection and the larger-batch
6-epoch recipe.

Training cost: ~5h per run (4 or 8 H100); strong-scaling at fixed global batch 32:
1/2/4/8/16 GPUs -> 100% / 98.4% / 92.5% / 76.3% / 54.5% efficiency (per-GPU
micro-batch 8 keeps 8 GPUs near-linear instead: 0.33s/step at global 64).
Full 15-dataset multi-vector eval (~1.2TB token embeddings): ~2.7h wall on the
sharded MaxSim pipeline.
