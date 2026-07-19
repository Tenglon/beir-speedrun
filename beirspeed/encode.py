"""Encode one corpus shard with mean-pooled BERT into fp16 embeddings.

Output per shard: {out_root}/{dataset}/shard_XXXX.npy (+ .ids.txt, .done).
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .data import iter_corpus


def load_model(model_dir, device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    return tok, model


@torch.no_grad()
def encode_texts(texts, tok, model, device, batch_size, max_length, log_every=0):
    """Mean-pooled embeddings, length-sorted batching, original order restored."""
    enc = tok(texts, truncation=True, max_length=max_length, padding=False)["input_ids"]
    order = sorted(range(len(texts)), key=lambda i: -len(enc[i]))
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    t0, done = time.time(), 0
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        batch = tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        out[idx] = emb.to(torch.float16).cpu().numpy()
        done += len(idx)
        if log_every and (b0 // batch_size) % log_every == 0:
            print("  %d/%d docs, %.0f docs/s" % (done, len(texts), done / (time.time() - t0)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "shard_%04d" % args.shard)
    if os.path.exists(base + ".done"):
        print("already done: %s" % base)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(args.model_dir, device)

    t0 = time.time()
    ids, texts = [], []
    for did, text in iter_corpus(args.data_root, args.dataset, args.start, args.end):
        ids.append(did)
        texts.append(text)
    print("read %d docs in %.0fs" % (len(ids), time.time() - t0), flush=True)

    t0 = time.time()
    emb = encode_texts(texts, tok, model, device, args.batch_size, args.max_length, log_every=50)
    print("encoded %d docs in %.0fs (%.0f docs/s)" % (
        len(ids), time.time() - t0, len(ids) / max(time.time() - t0, 1e-9)), flush=True)

    np.save(base + ".npy.tmp.npy", emb)
    os.replace(base + ".npy.tmp.npy", base + ".npy")
    with open(base + ".ids.txt.tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    os.replace(base + ".ids.txt.tmp", base + ".ids.txt")
    with open(base + ".done", "w") as f:
        f.write("%d\n" % len(ids))
    print("wrote %s.npy %s" % (base, emb.shape))


if __name__ == "__main__":
    main()
