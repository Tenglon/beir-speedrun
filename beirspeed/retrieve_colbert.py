"""Per-dataset ColBERT MaxSim retrieval over pre-encoded token-embedding shards.

Exact (brute-force) late interaction: for every doc chunk build a padded
[n, Lmax, dim] tensor, score sim[q,n] = sum_ql max_dl <q_ql, d_dl>, keep a
running top-k. Reuses the qrels/metrics/run-writing conventions of retrieve.py.
"""
import argparse
import glob
import gzip
import json
import os
import time

import numpy as np
import torch
from pylate import models

from .data import load_qrels, load_queries
from .halo import find_lift, load_colbert_with_lift
from .metrics import evaluate


def maxsim_scores(q, q_mask, d, d_mask, q_chunk=32, curv=None, mode="dot"):
    """q [Q,Lq,dim], d [n,Ld,dim] (cuda) -> scores [Q,n] fp32.

    mode "dot": dot-product MaxSim (fp16). "lorentz": negative Lorentz distance
    over lifted embeddings [x0, xs] in fp32. "hybrid" (HALO mainline):
    0.5 * spatial cosine + 0.5 * negative Lorentz distance."""
    out = torch.empty(q.shape[0], d.shape[0], device=q.device, dtype=torch.float32)
    if mode != "dot":
        q_chunk = max(q_chunk // 4, 4)
        d = d.float()
        d_norm = torch.nn.functional.normalize(d[..., 1:], dim=-1) if mode == "hybrid" else None
    neg = torch.finfo(torch.float16).min
    for b0 in range(0, q.shape[0], q_chunk):
        qb = q[b0:b0 + q_chunk]                                   # [b,Lq,dim]
        if mode == "dot":
            sim = torch.einsum("bld,ntd->bnlt", qb, d)            # [b,n,Lq,Ld]
        else:
            qb = qb.float()
            ip = torch.einsum("bld,ntd->bnlt", qb[..., 1:], d[..., 1:])
            ip = ip - qb[..., 0].unsqueeze(-1).unsqueeze(1) * d[..., 0][None, :, None, :]
            sim = -torch.acosh(torch.clamp(-curv * ip, min=1.0 + 1e-6)) / (curv ** 0.5)
            if mode == "hybrid":
                qn = torch.nn.functional.normalize(qb[..., 1:], dim=-1)
                sim = 0.5 * torch.einsum("bld,ntd->bnlt", qn, d_norm) + 0.5 * sim
        sim = sim.masked_fill(~d_mask[None, :, None, :], neg)
        best = sim.max(dim=-1).values.float()                     # [b,n,Lq]
        best = best * q_mask[b0:b0 + q_chunk][:, None, :]
        out[b0:b0 + q_chunk] = best.sum(dim=-1)
    return out


def load_shard(path):
    tokens = np.load(path)
    lens = np.load(path.replace(".tokens.npy", ".lens.npy"))
    with open(path.replace(".tokens.npy", ".ids.txt"), encoding="utf-8") as f:
        ids = f.read().splitlines()
    return tokens, lens, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--doc-chunk", type=int, default=4096)
    ap.add_argument("--expected-shards", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print("already done: %s" % out_dir)
        return

    shard_dir = os.path.join(args.emb_root, args.dataset)
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.tokens.npy")))
    done = [s for s in shards if os.path.exists(s.replace(".tokens.npy", ".done"))]
    if not done or (args.expected_shards and len(done) != args.expected_shards):
        raise SystemExit("incomplete shards for %s: %d done, expected %s" % (
            args.dataset, len(done), args.expected_shards or "?"))

    qrels = load_qrels(args.data_root, args.dataset, args.split)
    queries = load_queries(args.data_root, args.dataset, qids=set(qrels))
    qids = sorted(queries)
    print("%s: %d queries (%s), %d shards" % (args.dataset, len(qids), args.split, len(done)), flush=True)

    model = load_colbert_with_lift(args.model_dir).to("cuda").eval()
    lift = find_lift(model)
    curv = float(lift.curv()) if lift is not None else None
    mode = "dot" if lift is None else getattr(lift, "score_mode", "lorentz")
    if curv is not None:
        print("HALO retrieval: mode=%s curv=%.4f" % (mode, curv))
    q_embs = model.encode([queries[q] for q in qids], batch_size=256, is_query=True,
                          normalize_embeddings=lift is None,
                          convert_to_numpy=True, show_progress_bar=False)
    lq = max(e.shape[0] for e in q_embs)
    dim = q_embs[0].shape[1]
    q = torch.zeros(len(qids), lq, dim, dtype=torch.float16, device="cuda")
    q_mask = torch.zeros(len(qids), lq, dtype=torch.float32, device="cuda")
    for i, e in enumerate(q_embs):
        q[i, :e.shape[0]] = torch.from_numpy(e.astype(np.float16))
        q_mask[i, :e.shape[0]] = 1.0

    keep = args.k + 16
    top_scores = torch.full((len(qids), keep), -1e9, device="cuda", dtype=torch.float32)
    top_idx = torch.full((len(qids), keep), -1, device="cuda", dtype=torch.int64)
    all_ids, offset = [], 0
    t0 = time.time()
    for path in done:
        tokens, lens, ids = load_shard(path)
        starts = np.concatenate([[0], np.cumsum(lens)])
        for c0 in range(0, len(lens), args.doc_chunk):
            c1 = min(c0 + args.doc_chunk, len(lens))
            cl = torch.from_numpy(lens[c0:c1].astype(np.int64))
            flat = torch.from_numpy(tokens[starts[c0]:starts[c1]]).to("cuda")
            d = torch.nn.utils.rnn.pad_sequence(flat.split(cl.tolist()), batch_first=True)
            d_mask = (torch.arange(d.shape[1], device="cuda")[None, :] < cl.to("cuda")[:, None])
            scores = maxsim_scores(q, q_mask, d, d_mask, curv=curv, mode=mode)
            s, i = torch.topk(scores, min(keep, scores.shape[1]), dim=1)
            top_scores = torch.cat([top_scores, s], dim=1)
            top_idx = torch.cat([top_idx, i + offset + c0], dim=1)
            top_scores, pos2 = torch.topk(top_scores, keep, dim=1)
            top_idx = torch.gather(top_idx, 1, pos2)
        all_ids.extend(ids)
        offset += len(lens)
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
        "config": {"model": args.model_dir.rstrip("/").split("/")[-1],
                   "scoring": "colbert_maxsim",
                   "similarity": {"dot": "dot", "lorentz": "neg_lorentz_distance",
                                  "hybrid": "hybrid_cos_neg_lorentz"}[mode],
                   "curv": curv},
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
