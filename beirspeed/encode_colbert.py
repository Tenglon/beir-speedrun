"""Encode one corpus shard with a PyLate ColBERT into per-token fp16 embeddings.

Output per shard: shard_XXXX.tokens.npy [total_tokens, dim] fp16,
shard_XXXX.lens.npy int32 doc lengths, shard_XXXX.ids.txt, shard_XXXX.done.
Padding/punctuation tokens are already stripped by PyLate.
"""
import argparse
import os
import time

import numpy as np
import torch
from pylate import models

from .data import iter_corpus
from .halo import find_lift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--doc-length", type=int, default=300)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "shard_%04d" % args.shard)
    if os.path.exists(base + ".done"):
        print("already done: %s" % base)
        return

    model = models.ColBERT(model_name_or_path=args.model_dir, document_length=args.doc_length)
    model = model.to("cuda").eval()
    lift = find_lift(model)
    normalize = lift is None  # hyperbolic embeddings live on the hyperboloid, not the sphere
    if lift is not None:
        print("HALO lift detected: curv=%.4f, storing lifted (dim+1) embeddings" % lift.curv().item())

    t0 = time.time()
    ids, texts = [], []
    for did, text in iter_corpus(args.data_root, args.dataset, args.start, args.end):
        ids.append(did)
        texts.append(text)
    print("read %d docs in %.0fs" % (len(ids), time.time() - t0), flush=True)

    t0 = time.time()
    embs = model.encode(texts, batch_size=args.batch_size, is_query=False,
                        normalize_embeddings=normalize,
                        convert_to_numpy=True, show_progress_bar=False)
    lens = np.array([e.shape[0] for e in embs], dtype=np.int32)
    tokens = np.concatenate(embs, axis=0).astype(np.float16)
    dt = time.time() - t0
    print("encoded %d docs -> %d tokens in %.0fs (%.0f docs/s)" % (
        len(ids), tokens.shape[0], dt, len(ids) / max(dt, 1e-9)), flush=True)

    np.save(base + ".tokens.tmp.npy", tokens)
    os.replace(base + ".tokens.tmp.npy", base + ".tokens.npy")
    np.save(base + ".lens.tmp.npy", lens)
    os.replace(base + ".lens.tmp.npy", base + ".lens.npy")
    with open(base + ".ids.txt.tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    os.replace(base + ".ids.txt.tmp", base + ".ids.txt")
    with open(base + ".done", "w") as f:
        f.write("%d\n" % len(ids))
    print("wrote %s.tokens.npy %s" % (base, tokens.shape))


if __name__ == "__main__":
    main()
