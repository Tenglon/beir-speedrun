"""Per-dataset retrieval + evaluation over pre-encoded corpus shards.

Encodes queries, streams shard embeddings, keeps a running cosine top-k,
drops self-matches (doc_id == query_id), writes metrics.json + run.tsv.gz.
"""
import argparse
import glob
import gzip
import json
import os
import time

import numpy as np
import torch

from .data import load_qrels, load_queries
from .encode import encode_texts, load_model
from .metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--doc-chunk", type=int, default=131072)
    ap.add_argument("--expected-shards", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print("already done: %s" % out_dir)
        return

    shard_dir = os.path.join(args.emb_root, args.dataset)
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npy")))
    done = [s for s in shards if os.path.exists(s.replace(".npy", ".done"))]
    if not done or (args.expected_shards and len(done) != args.expected_shards):
        raise SystemExit("incomplete shards for %s: %d done, expected %s" % (
            args.dataset, len(done), args.expected_shards or "?"))

    qrels = load_qrels(args.data_root, args.dataset, args.split)
    queries = load_queries(args.data_root, args.dataset, qids=set(qrels))
    qids = sorted(queries)
    print("%s: %d queries (%s), %d shards" % (args.dataset, len(qids), args.split, len(done)), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(args.model_dir, device)
    q_emb = encode_texts([queries[q] for q in qids], tok, model, device, 256, args.max_length)
    q = torch.from_numpy(q_emb).to(device).float()
    q = torch.nn.functional.normalize(q, dim=-1).to(torch.float16)

    keep = args.k + 16  # headroom for dropping self-matches
    top_scores = torch.full((len(qids), keep), -2.0, device=device, dtype=torch.float32)
    top_idx = torch.full((len(qids), keep), -1, device=device, dtype=torch.int64)
    all_ids, offset = [], 0
    t0 = time.time()
    for path in done:
        with open(path.replace(".npy", ".ids.txt"), encoding="utf-8") as f:
            ids = f.read().splitlines()
        d = torch.from_numpy(np.load(path)).to(device)
        d = torch.nn.functional.normalize(d.float(), dim=-1).to(torch.float16)
        for c0 in range(0, d.shape[0], args.doc_chunk):
            chunk = d[c0:c0 + args.doc_chunk]
            scores = (q @ chunk.T).float()
            s, i = torch.topk(scores, min(keep, chunk.shape[0]), dim=1)
            top_scores = torch.cat([top_scores, s], dim=1)
            top_idx = torch.cat([top_idx, i + offset + c0], dim=1)
            top_scores, pos = torch.topk(top_scores, keep, dim=1)
            top_idx = torch.gather(top_idx, 1, pos)
        all_ids.extend(ids)
        offset += d.shape[0]
        print("  searched %d docs (%.0fs)" % (offset, time.time() - t0), flush=True)

    top_scores = top_scores.cpu().numpy()
    top_idx = top_idx.cpu().numpy()
    run = {}
    for qi, qid in enumerate(qids):
        docs = {}
        for s, di in zip(top_scores[qi], top_idx[qi]):
            if di < 0 or len(docs) >= args.k:
                continue
            did = all_ids[di]
            if did != qid:  # BEIR convention: ignore identical ids
                docs[did] = float(s)
        run[qid] = docs

    means, backend = evaluate(qrels, run)
    result = {
        "dataset": args.dataset, "split": args.split, "num_queries": len(qids),
        "num_docs": offset, "num_shards": len(done), "k": args.k,
        "metrics": {name: round(val, 5) for name, val in sorted(means.items())},
        "eval_backend": backend,
        "config": {"model": os.path.basename(args.model_dir.rstrip("/")),
                   "pooling": "mean", "similarity": "cosine",
                   "max_length": args.max_length, "zero_shot": True},
    }
    with gzip.open(os.path.join(out_dir, "run.tsv.gz"), "wt", encoding="utf-8") as f:
        for qid in qids:
            for rank, (did, s) in enumerate(sorted(run[qid].items(), key=lambda kv: -kv[1]), 1):
                f.write("%s\t%s\t%d\t%.5f\n" % (qid, did, rank, s))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["metrics"], indent=1))


if __name__ == "__main__":
    main()
