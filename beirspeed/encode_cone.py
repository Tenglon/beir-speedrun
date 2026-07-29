"""Encode one corpus shard with ConeColBERT: euclid tokens + cone points."""
import argparse
import os
import time

import numpy as np
import torch
from pylate import models

from conecolbert.modeling import ConeColBERT
from .data import iter_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--colbert-dir", required=True)
    ap.add_argument("--cone-pt", required=True)
    ap.add_argument("--batch-size", type=int, default=192)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "shard_%04d" % args.shard)
    if os.path.exists(base + ".done"):
        print("already done", base)
        return

    colbert = models.ColBERT(model_name_or_path=args.colbert_dir, device="cuda")
    model = ConeColBERT(colbert).to("cuda").eval()
    st = torch.load(args.cone_pt, map_location="cpu", weights_only=False)
    model.hyper_head.load_state_dict(st["hyper_head"])
    model.gate.load_state_dict(st["gate"])
    with torch.no_grad():
        model.scale_raw.fill_(float(st["scale_raw"]))
        model.v0_raw.fill_(float(st["v0_raw"]))

    ids, texts = [], []
    for did, text in iter_corpus(args.data_root, args.dataset, args.start, args.end):
        ids.append(did)
        texts.append(text)
    t0 = time.time()
    E, P, L = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            f = colbert.tokenize(texts[i:i + args.batch_size], is_query=False)
            f = {k: v.to("cuda") for k, v in f.items()}
            e, _, pts, _, valid, _ = model._encode(f, "document")
            for j in range(e.shape[0]):
                m = valid[j]
                E.append(e[j][m].half().cpu().numpy())
                P.append(pts[j][m].half().cpu().numpy())
                L.append(int(m.sum()))
    print("encoded %d docs in %.0fs" % (len(ids), time.time() - t0), flush=True)
    np.save(base + ".tokens.tmp.npy", np.concatenate(E))
    os.replace(base + ".tokens.tmp.npy", base + ".tokens.npy")
    np.save(base + ".points.tmp.npy", np.concatenate(P))
    os.replace(base + ".points.tmp.npy", base + ".points.npy")
    np.save(base + ".lens.tmp.npy", np.array(L, dtype=np.int32))
    os.replace(base + ".lens.tmp.npy", base + ".lens.npy")
    with open(base + ".ids.txt.tmp", "w") as f:
        f.write("\n".join(ids) + "\n")
    os.replace(base + ".ids.txt.tmp", base + ".ids.txt")
    open(base + ".done", "w").write("%d\n" % len(ids))
    print("wrote", base)


if __name__ == "__main__":
    main()
