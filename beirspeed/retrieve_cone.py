"""Per-dataset fused (euclid + cone) MaxSim retrieval over cone shards.

Writes BOTH fused metrics.json and euclid-only metrics_euclid.json so every
dataset carries its own unhandicapped control."""
import argparse
import glob
import gzip
import json
import os
import time

import numpy as np
import torch
from pylate import models

from conecolbert.geometry import cone_violation
from conecolbert.modeling import ConeColBERT
from .data import load_qrels, load_queries
from .metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--colbert-dir", required=True)
    ap.add_argument("--cone-pt", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--doc-chunk", type=int, default=1024)
    ap.add_argument("--q-chunk", type=int, default=8)
    ap.add_argument("--expected-shards", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print("already done", out_dir)
        return
    shards = sorted(glob.glob(os.path.join(args.emb_root, args.dataset, "shard_*.tokens.npy")))
    done = [s for s in shards if os.path.exists(s.replace(".tokens.npy", ".done"))]
    if not done or (args.expected_shards and len(done) != args.expected_shards):
        raise SystemExit("incomplete shards: %d/%s" % (len(done), args.expected_shards))

    qrels = load_qrels(args.data_root, args.dataset, args.split)
    queries = load_queries(args.data_root, args.dataset, qids=set(qrels))
    qids = sorted(queries)

    colbert = models.ColBERT(model_name_or_path=args.colbert_dir, device="cuda")
    model = ConeColBERT(colbert).to("cuda").eval()
    st = torch.load(args.cone_pt, map_location="cpu", weights_only=False)
    model.hyper_head.load_state_dict(st["hyper_head"])
    model.gate.load_state_dict(st["gate"])
    with torch.no_grad():
        model.scale_raw.fill_(float(st["scale_raw"]))
        model.v0_raw.fill_(float(st["v0_raw"]))
    scale, v0 = model.hyperbolic_scale().item(), model.v0().item()
    print("%s: %d q, %d shards, scale=%.3f v0=%.4f" % (
        args.dataset, len(qids), len(done), scale, v0), flush=True)

    qE, qP, qG = [], [], []
    with torch.no_grad():
        for i in range(0, len(qids), 256):
            f = colbert.tokenize([queries[q] for q in qids[i:i + 256]], is_query=True)
            f = {k: v.to("cuda") for k, v in f.items()}
            e, _, pts, _, _, g = model._encode(f, "query")
            qE.append(e.float()); qP.append(pts.float()); qG.append(g.float())
    qE, qP, qG = torch.cat(qE), torch.cat(qP), torch.cat(qG)

    keep = args.k + 16
    tops_f = torch.full((len(qids), keep), -1e9, device="cuda")
    topi_f = torch.full((len(qids), keep), -1, dtype=torch.int64, device="cuda")
    tops_e = tops_f.clone(); topi_e = topi_f.clone()
    all_ids, offset = [], 0
    t0 = time.time()
    for path in done:
        tok = np.load(path); pts = np.load(path.replace(".tokens.npy", ".points.npy"))
        lens = np.load(path.replace(".tokens.npy", ".lens.npy"))
        with open(path.replace(".tokens.npy", ".ids.txt")) as f:
            ids = f.read().splitlines()
        starts = np.concatenate([[0], np.cumsum(lens)])
        for c0 in range(0, len(lens), args.doc_chunk):
            c1 = min(c0 + args.doc_chunk, len(lens))
            cl = torch.from_numpy(lens[c0:c1].astype(np.int64))
            de = torch.from_numpy(tok[starts[c0]:starts[c1]]).cuda().float()
            dp = torch.from_numpy(pts[starts[c0]:starts[c1]]).cuda().float()
            de = torch.nn.utils.rnn.pad_sequence(de.split(cl.tolist()), batch_first=True)
            dp = torch.nn.utils.rnn.pad_sequence(dp.split(cl.tolist()), batch_first=True)
            dm = (torch.arange(de.shape[1], device="cuda")[None] < cl.cuda()[:, None])
            for b0 in range(0, len(qids), args.q_chunk):
                qe = qE[b0:b0 + args.q_chunk].cuda()
                qp = qP[b0:b0 + args.q_chunk].cuda()
                qg = qG[b0:b0 + args.q_chunk].cuda()
                e = torch.einsum("qlh,dth->qdlt", qe, de)
                v = cone_violation(
                    qp.unsqueeze(1).expand(-1, de.shape[0], -1, -1).flatten(0, 1),
                    dp.unsqueeze(0).expand(qe.shape[0], -1, -1, -1).flatten(0, 1),
                    document_mask=dm.unsqueeze(0).expand(qe.shape[0], -1, -1).flatten(0, 1),
                ).view(qe.shape[0], de.shape[0], qe.shape[1], -1)
                m = e - scale * qg.unsqueeze(1).unsqueeze(-1) * (v - v0).clamp_min(0)
                neg = torch.finfo(torch.float32).min
                for s, tops, topi in [(e, tops_e, topi_e), (m, tops_f, topi_f)]:
                    sc = s.masked_fill(~dm[None, :, None, :], neg).max(-1).values.sum(-1)
                    tv, ti = sc.topk(min(keep, sc.shape[1]), dim=1)
                    cat_s = torch.cat([tops[b0:b0 + args.q_chunk], tv], 1)
                    cat_i = torch.cat([topi[b0:b0 + args.q_chunk], ti + offset + c0], 1)
                    ns, pos = cat_s.topk(keep, dim=1)
                    tops[b0:b0 + args.q_chunk] = ns
                    topi[b0:b0 + args.q_chunk] = torch.gather(cat_i, 1, pos)
        all_ids += ids
        offset += len(lens)
        print("  searched %d docs (%.0fs)" % (offset, time.time() - t0), flush=True)

    for tag, tops, topi in [("", tops_f, topi_f), ("_euclid", tops_e, topi_e)]:
        run = {}
        ts, ti = tops.cpu().numpy(), topi.cpu().numpy()
        for qi, qid in enumerate(qids):
            docs = {}
            for s, di in zip(ts[qi], ti[qi]):
                if di >= 0 and len(docs) < args.k and all_ids[di] != qid:
                    docs[all_ids[di]] = float(s)
            run[qid] = docs
        means, backend = evaluate(qrels, run)
        result = {"dataset": args.dataset, "split": args.split, "num_queries": len(qids),
                  "num_docs": offset,
                  "metrics": {k2: round(v2, 5) for k2, v2 in sorted(means.items())},
                  "config": {"fusion": "euclid" if tag else "cone_fused",
                             "scale": scale, "v0": v0}}
        with open(os.path.join(out_dir, "metrics%s.json" % tag), "w") as f:
            json.dump(result, f, indent=1)
        print(tag or "fused", json.dumps(result["metrics"]), flush=True)


if __name__ == "__main__":
    main()
