"""Build a global shard manifest: every dataset split into equal-size shards.

Usage:
  python -m beirspeed.shard --data-root D --out manifest.json --shard-size 500000
  python -m beirspeed.shard --manifest manifest.json --task 17 --print-env
"""
import argparse
import json
import os

from .data import corpus_path, count_lines


def discover_datasets(data_root):
    names = []
    for root, dirs, files in os.walk(data_root):
        if "corpus.jsonl" in files:
            names.append(os.path.relpath(root, data_root))
            dirs.clear()
    return sorted(names)


def build_manifest(data_root, shard_size):
    tasks = []
    counts = {}
    for ds in discover_datasets(data_root):
        n = count_lines(corpus_path(data_root, ds))
        counts[ds] = n
        for shard, start in enumerate(range(0, n, shard_size)):
            tasks.append({
                "task": len(tasks),
                "dataset": ds,
                "shard": shard,
                "start": start,
                "end": min(start + shard_size, n),
            })
    return {"shard_size": shard_size, "corpus_counts": counts, "tasks": tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root")
    ap.add_argument("--out")
    ap.add_argument("--shard-size", type=int, default=500000)
    ap.add_argument("--manifest")
    ap.add_argument("--task", type=int)
    ap.add_argument("--print-env", action="store_true")
    args = ap.parse_args()

    if args.print_env:
        with open(args.manifest) as f:
            t = json.load(f)["tasks"][args.task]
        print("DS=%s START=%d END=%d SHARD=%d" % (t["dataset"], t["start"], t["end"], t["shard"]))
        return

    m = build_manifest(args.data_root, args.shard_size)
    with open(args.out, "w") as f:
        json.dump(m, f, indent=1)
    per_ds = {}
    for t in m["tasks"]:
        per_ds[t["dataset"]] = per_ds.get(t["dataset"], 0) + 1
    print("datasets=%d docs=%d shards=%d" % (
        len(m["corpus_counts"]), sum(m["corpus_counts"].values()), len(m["tasks"])))
    for ds in sorted(per_ds):
        print("  %-28s %9d docs %3d shards" % (ds, m["corpus_counts"][ds], per_ds[ds]))


if __name__ == "__main__":
    main()
