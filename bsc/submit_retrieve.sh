#!/bin/bash
# Build retrieve_targets.tsv (dataset, split, expected shard count) and submit the
# retrieval array: one single-GPU task per dataset. msmarco evaluates on dev.
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$VENV/bin/python" - <<'PY'
import json, os
root = os.environ["EMB_ROOT"]
m = json.load(open(os.path.join(root, "manifest.json")))
shards = {}
for t in m["tasks"]:
    shards[t["dataset"]] = shards.get(t["dataset"], 0) + 1
with open(os.path.join(root, "retrieve_targets.tsv"), "w") as f:
    for ds in sorted(shards):
        split = "dev" if ds == "msmarco" else "test"
        f.write("%s\t%s\t%d\n" % (ds, split, shards[ds]))
print("targets:", len(shards))
PY

N=$(wc -l < "$EMB_ROOT/retrieve_targets.tsv")
echo "submitting retrieve array 0-$((N - 1))"
sbatch --array="${1:-0-$((N - 1))}%${THROTTLE:-100}" ${SBATCH_EXTRA:-} "$CODE_ROOT/bsc/retrieve_array.sbatch"
