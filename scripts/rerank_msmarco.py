"""In-domain decisive test: rerank run2's msmarco-dev top-100 with ConeColBERT.

Compares euclid-only vs fused MRR@10/nDCG@10 on the same candidate sets.
"""
import argparse
import gzip
import json
import os

import torch
from pylate import models

from beirspeed.data import load_qrels, load_queries, corpus_path
from beirspeed.metrics import evaluate
from conecolbert.geometry import cone_violation
from conecolbert.modeling import ConeColBERT

ROOT = os.environ.get("BEIR_ROOT", "/gpfs/scratch/ehpc821/uoa994647/beir_speedrun")


@torch.no_grad()
def encode(model, texts, is_query, bs=256):
    E, P, M, G = [], [], [], []
    for i in range(0, len(texts), bs):
        f = model.colbert.tokenize(texts[i:i + bs], is_query=is_query)
        f = {k: v.to("cuda") for k, v in f.items()}
        e, _, pts, _, valid, gates = model._encode(f, "query" if is_query else "document")
        E += [t for t in e.half().cpu().unbind(0)]
        P += [t for t in pts.half().cpu().unbind(0)]
        M += [t for t in valid.cpu().unbind(0)]
        if is_query:
            G += [t for t in gates.float().cpu().unbind(0)]
    return E, P, M, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ROOT + "/colbert/run2_8gpu/checkpoint-60000")
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--run", default=ROOT + "/results/msmarco/run.tsv.gz")
    ap.add_argument("--max-queries", type=int, default=2000)
    args = ap.parse_args()

    cand = {}
    with gzip.open(args.run, "rt") as f:
        for line in f:
            qid, did = line.split("\t")[:2]
            cand.setdefault(qid, []).append(did)
    qrels = load_qrels(ROOT + "/datasets", "msmarco", "dev")
    qids = sorted(set(cand) & set(qrels))[:args.max_queries]
    need = sorted({d for q in qids for d in cand[q]})
    print("queries=%d unique_docs=%d" % (len(qids), len(need)), flush=True)

    texts = {}
    with open(corpus_path(ROOT + "/datasets", "msmarco"), encoding="utf-8") as f:
        needset = set(need)
        for line in f:
            d = json.loads(line)
            if str(d["_id"]) in needset:
                t = ((d.get("title") or "").strip() + " " + (d.get("text") or "").strip()).strip()
                texts[str(d["_id"])] = t
                if len(texts) == len(needset):
                    break

    colbert = models.ColBERT(model_name_or_path=args.ckpt, device="cuda")
    model = ConeColBERT(colbert).to("cuda").eval()
    st = torch.load(args.phase2, map_location="cpu", weights_only=False)
    model.hyper_head.load_state_dict(st["hyper_head"])
    model.gate.load_state_dict(st["gate"])
    with torch.no_grad():
        model.scale_raw.fill_(float(st["scale_raw"]))
        if "v0_raw" in st:
            model.v0_raw.fill_(float(st["v0_raw"]))
    scale = model.hyperbolic_scale().item()
    print("scale=%.3f" % scale, flush=True)

    queries = load_queries(ROOT + "/datasets", "msmarco", qids=set(qids))
    qE, qP, _, qG = encode(model, [queries[q] for q in qids], True)
    doc_order = need
    dE, dP, dM, _ = encode(model, [texts[d] for d in doc_order], False)
    dpos = {d: i for i, d in enumerate(doc_order)}

    run_e, run_f = {}, {}
    for i, qid in enumerate(qids):
        qe = qE[i].float().cuda(); qp = qP[i].float().cuda(); qg = qG[i].float().cuda()
        idx = [dpos[d] for d in cand[qid]]
        de = torch.nn.utils.rnn.pad_sequence([dE[j].float() for j in idx], batch_first=True).cuda()
        dp = torch.nn.utils.rnn.pad_sequence([dP[j].float() for j in idx], batch_first=True).cuda()
        dm = torch.nn.utils.rnn.pad_sequence([dM[j] for j in idx], batch_first=True).cuda()
        e = torch.einsum("lh,dth->dlt", qe, de)
        v = cone_violation(qp.unsqueeze(0).expand(de.shape[0], -1, -1), dp,
                           document_mask=dm)
        m = e - scale * qg.unsqueeze(0).unsqueeze(-1) * (v - model.v0().item()).clamp_min(0)
        neg = torch.finfo(torch.float32).min
        se = e.masked_fill(~dm.unsqueeze(1), neg).max(-1).values.sum(-1)
        sf = m.masked_fill(~dm.unsqueeze(1), neg).max(-1).values.sum(-1)
        run_e[qid] = {d: float(s) for d, s in zip(cand[qid], se)}
        run_f[qid] = {d: float(s) for d, s in zip(cand[qid], sf)}
        if i % 500 == 0:
            print("reranked", i, flush=True)

    sub_qrels = {q: qrels[q] for q in qids}
    for name, run in [("euclid", run_e), ("fused", run_f)]:
        means, _ = evaluate(sub_qrels, run)
        print(name, json.dumps({k: round(v, 4) for k, v in means.items()}), flush=True)


if __name__ == "__main__":
    main()
