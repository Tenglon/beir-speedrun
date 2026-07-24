"""Small-corpus ConeColBERT evaluation: euclidean-only vs fused, one GPU job.

python scripts/eval_cone.py --datasets scifact nfcorpus fiqa --phase2 .../cone_phase2.pt
"""
import argparse
import json
import os

import torch
from pylate import models

from beirspeed.data import iter_corpus, load_qrels, load_queries
from beirspeed.metrics import evaluate
from conecolbert.geometry import cone_violation
from conecolbert.modeling import ConeColBERT

ROOT = os.environ.get("BEIR_ROOT", "/gpfs/scratch/ehpc821/uoa994647/beir_speedrun")


@torch.no_grad()
def encode(model, texts, is_query, bs=128):
    E, P, M, G = [], [], [], []
    tok = model.colbert.tokenize
    for i in range(0, len(texts), bs):
        f = tok(texts[i:i + bs], is_query=is_query)
        f = {k: v.to("cuda") for k, v in f.items()}
        role = "query" if is_query else "document"
        e, _, pts, _, valid, gates = model._encode(f, role)
        E.append(e.half().cpu()); P.append(pts.half().cpu()); M.append(valid.cpu())
        if is_query:
            G.append(gates.float().cpu())
    pad = lambda ts: torch.nn.utils.rnn.pad_sequence(
        [t for b in ts for t in b.unbind(0)], batch_first=True)
    return pad(E), pad(P), pad(M), (pad(G) if is_query else None)


@torch.no_grad()
def search(qE, qP, qG, dE, dP, dM, scale, k=100, chunk=1024, q_chunk=8):
    n = dE.shape[0]
    outs = []
    for q0 in range(0, qE.shape[0], q_chunk):
        qe = qE[q0:q0 + q_chunk].float().cuda()
        qp = qP[q0:q0 + q_chunk].float().cuda()
        qg = qG[q0:q0 + q_chunk].float().cuda()
        se_l, sf_l = [], []
        for c0 in range(0, n, chunk):
            de = dE[c0:c0 + chunk].float().cuda()
            dp = dP[c0:c0 + chunk].float().cuda()
            dm = dM[c0:c0 + chunk].cuda()
            e = torch.einsum("qlh,dth->qdlt", qe, de)
            v = cone_violation(qp.unsqueeze(1).expand(-1, de.shape[0], -1, -1).flatten(0, 1),
                               dp.unsqueeze(0).expand(qe.shape[0], -1, -1, -1).flatten(0, 1),
                               document_mask=dm.unsqueeze(0).expand(qe.shape[0], -1, -1).flatten(0, 1)
                               ).view(qe.shape[0], de.shape[0], qe.shape[1], -1)
            m = e - scale * qg.unsqueeze(1).unsqueeze(-1) * v
            neg = torch.finfo(torch.float32).min
            se_l.append(e.masked_fill(~dm[None, :, None, :], neg).max(-1).values.sum(-1))
            sf_l.append(m.masked_fill(~dm[None, :, None, :], neg).max(-1).values.sum(-1))
        outs.append((torch.cat(se_l, 1).cpu(), torch.cat(sf_l, 1).cpu()))
    se = torch.cat([a for a, _ in outs]); sf = torch.cat([b for _, b in outs])
    return se, sf


def run_dataset(model, scale, ds, out):
    qrels = load_qrels(ROOT + "/datasets", ds, "test")
    queries = load_queries(ROOT + "/datasets", ds, qids=set(qrels))
    qids = sorted(queries)
    ids, texts = zip(*list(iter_corpus(ROOT + "/datasets", ds, 0, 10**9)))
    qE, qP, _, qG = encode(model, [queries[q] for q in qids], True)
    dE, dP, dM, _ = encode(model, list(texts), False)
    se, sf, = search(qE, qP, qG, dE, dP, dM, scale)
    res = {}
    for name, s in [("euclid", se), ("fused", sf)]:
        topv, topi = s.topk(min(100, s.shape[1]), dim=1)
        run = {qid: {ids[di]: float(sv) for sv, di in zip(topv[i], topi[i])
                     if ids[di] != qid}
               for i, qid in enumerate(qids)}
        means, _ = evaluate(qrels, run)
        res[name] = round(means["ndcg_cut_10"], 4)
    print(ds, json.dumps(res), flush=True)
    out[ds] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ROOT + "/colbert/run2_8gpu/checkpoint-60000")
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    args = ap.parse_args()

    colbert = models.ColBERT(model_name_or_path=args.ckpt, device="cuda")
    model = ConeColBERT(colbert).to("cuda").eval()
    state = torch.load(args.phase2, map_location="cpu", weights_only=False)
    model.hyper_head.load_state_dict(state["hyper_head"])
    model.gate.load_state_dict(state["gate"])
    with torch.no_grad():
        model.scale_raw.fill_(float(state["scale_raw"]))
    scale = model.hyperbolic_scale().item()
    print("loaded phase2: scale=%.3f" % scale, flush=True)

    out = {}
    for ds in args.datasets:
        run_dataset(model, scale, ds, out)
    print("SUMMARY", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
