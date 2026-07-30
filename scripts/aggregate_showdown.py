"""Final showdown table: ConeColBERT fused vs its own euclid control vs run2.

python scripts/aggregate_showdown.py --results .../joint60k/results \
    --run2 results/colbert/results_table.json --out showdown.md
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def collect(results_root, fname):
    per = {}
    for f in glob.glob(os.path.join(results_root, "**", fname), recursive=True):
        d = json.load(open(f))
        per[d["dataset"]] = d["metrics"]["ndcg_cut_10"]
    # cqadupstack sub-forums -> single averaged entry
    cqa = [k for k in per if k.startswith("cqadupstack/")]
    if cqa:
        per["cqadupstack"] = sum(per[k] for k in cqa) / len(cqa)
        for k in cqa:
            del per[k]
    return per


ORDER = ["msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana",
         "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs",
         "fever", "climate-fever", "scifact"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    fused = collect(args.results, "metrics.json")
    euclid = collect(args.results, "metrics_euclid.json")
    raw = json.load(open(args.run2))
    run2 = {}
    for k2, v2 in raw.items():
        if k2.startswith("cqadupstack/"):
            run2.setdefault("cqadupstack", []).append(v2["metrics"]["ndcg_cut_10"])
        else:
            run2[k2] = v2["metrics"]["ndcg_cut_10"]
    if isinstance(run2.get("cqadupstack"), list):
        run2["cqadupstack"] = sum(run2["cqadupstack"]) / len(run2["cqadupstack"])

    lines = ["| dataset | run2 (euclid SOTA) | joint euclid | joint fused | fused-run2 | fused-euclid |",
             "|---|---:|---:|---:|---:|---:|"]
    sums = defaultdict(float)
    n = 0
    for ds in ORDER:
        if ds not in fused:
            continue
        r2, eu, fu = run2.get(ds, float("nan")), euclid.get(ds, 0), fused[ds]
        lines.append("| %s | %.4f | %.4f | %.4f | %+.4f | %+.4f |" % (
            ds, r2, eu, fu, fu - r2, fu - eu))
        if ds != "msmarco":
            sums["r2"] += r2; sums["eu"] += eu; sums["fu"] += fu; n += 1
    if n:
        lines.append("| **avg (%d, excl msmarco)** | **%.4f** | **%.4f** | **%.4f** | **%+.4f** | **%+.4f** |" % (
            n, sums["r2"] / n, sums["eu"] / n, sums["fu"] / n,
            (sums["fu"] - sums["r2"]) / n, (sums["fu"] - sums["eu"]) / n))
    table = "\n".join(lines)
    print(table)
    if args.out:
        open(args.out, "w").write(table + "\n")


if __name__ == "__main__":
    main()
