"""trec_eval-compatible retrieval metrics (pytrec_eval when available)."""
import math


def _ranked(run_q):
    return [d for d, _ in sorted(run_q.items(), key=lambda kv: -kv[1])]


def _ndcg(qrels_q, ranking, k):
    # trec_eval ndcg_cut: linear gain, discount 1/log2(max(rank,2))
    dcg = sum(qrels_q.get(d, 0) / math.log2(max(i + 1, 2)) for i, d in enumerate(ranking[:k]))
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(max(i + 1, 2)) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _fallback(qrels, run, ndcg_ks, recall_ks):
    per_q = {}
    for qid, qrels_q in qrels.items():
        ranking = _ranked(run.get(qid, {}))
        rel = {d for d, r in qrels_q.items() if r > 0}
        m = {}
        for k in ndcg_ks:
            m["ndcg_cut_%d" % k] = _ndcg(qrels_q, ranking, k)
        for k in recall_ks:
            m["recall_%d" % k] = (len(rel & set(ranking[:k])) / len(rel)) if rel else 0.0
        per_q[qid] = m
    return per_q


def evaluate(qrels, run, ndcg_ks=(10, 100), recall_ks=(100,)):
    """Mean metrics over queries in qrels. Adds mrr@10. Returns (means, backend)."""
    try:
        import pytrec_eval
        measures = {"ndcg_cut." + ",".join(map(str, ndcg_ks)),
                    "recall." + ",".join(map(str, recall_ks))}
        ev = pytrec_eval.RelevanceEvaluator(qrels, measures)
        per_q = ev.evaluate({q: dict(d) for q, d in run.items()})
        backend = "pytrec_eval"
    except ImportError:
        per_q = _fallback(qrels, run, ndcg_ks, recall_ks)
        backend = "fallback"

    means = {}
    for qid in qrels:
        for name, val in per_q.get(qid, {}).items():
            means[name] = means.get(name, 0.0) + val
    n = max(len(qrels), 1)
    means = {name: val / n for name, val in means.items()}

    mrr = 0.0
    for qid, qrels_q in qrels.items():
        rel = {d for d, r in qrels_q.items() if r > 0}
        for i, d in enumerate(_ranked(run.get(qid, {}))[:10]):
            if d in rel:
                mrr += 1.0 / (i + 1)
                break
    means["mrr_10"] = mrr / n
    return means, backend
