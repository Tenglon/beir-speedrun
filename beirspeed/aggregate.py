"""Aggregate per-dataset metrics.json into the final BEIR table.

cqadupstack/* sub-forums are averaged into a single row (BEIR convention).
"""
import argparse
import glob
import json
import os

ORDER = ["msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana",
         "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs",
         "fever", "climate-fever", "scifact"]
COLS = ["ndcg_cut_10", "ndcg_cut_100", "recall_100", "mrr_10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = {}
    cqa = []
    for path in sorted(glob.glob(os.path.join(args.results_root, "**", "metrics.json"), recursive=True)):
        with open(path) as f:
            r = json.load(f)
        if r["dataset"].startswith("cqadupstack/"):
            cqa.append(r)
        else:
            rows[r["dataset"]] = r
    if cqa:
        rows["cqadupstack"] = {
            "dataset": "cqadupstack", "split": "test",
            "num_queries": sum(r["num_queries"] for r in cqa),
            "num_docs": sum(r["num_docs"] for r in cqa),
            "metrics": {c: sum(r["metrics"][c] for r in cqa) / len(cqa) for c in COLS},
            "sub_datasets": len(cqa),
        }

    names = [d for d in ORDER if d in rows] + sorted(set(rows) - set(ORDER))
    lines = ["| dataset | split | queries | docs | " + " | ".join(c for c in COLS) + " |",
             "|---|---|---:|---:|" + "---:|" * len(COLS)]
    for d in names:
        r = rows[d]
        lines.append("| %s | %s | %d | %d | %s |" % (
            d, r["split"], r["num_queries"], r["num_docs"],
            " | ".join("%.4f" % r["metrics"][c] for c in COLS)))
    beir14 = [d for d in names if d != "msmarco"]
    for label, group in [("avg (14 BEIR, excl msmarco)", beir14), ("avg (all %d)" % len(names), names)]:
        if group:
            lines.append("| **%s** | | | | %s |" % (
                label, " | ".join("%.4f" % (sum(rows[d]["metrics"][c] for d in group) / len(group))
                                  for c in COLS)))
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        with open(args.out.replace(".md", ".json"), "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
