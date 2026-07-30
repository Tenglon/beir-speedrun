#!/bin/bash
# Evaluate a ColBERT checkpoint on BEIR: encode array + dependent retrieve array.
# Usage: submit_colbert_eval.sh <tag> <checkpoint_dir> [small6|all]
# small6 = scifact nfcorpus arguana scidocs fiqa trec-covid (fast per-epoch eval).
set -euo pipefail
source "$(dirname "$0")/env.sh"

TAG=$1
export CB_MODEL_DIR=$2
SCOPE=${3:-small6}
export CB_EMB_ROOT=$BEIR_ROOT/colbert/$TAG/embeddings
export CB_RESULTS_ROOT=$BEIR_ROOT/colbert/$TAG/results
mkdir -p "$CB_EMB_ROOT" "$CB_RESULTS_ROOT"

SMALL6="arguana fiqa nfcorpus scidocs scifact trec-covid"
readarray -t SEL < <(SCOPE_DS="$([ "$SCOPE" = small6 ] && echo "$SMALL6" || echo "")" \
  "$VENV/bin/python" - <<'PY'
import json, os
m = json.load(open(os.path.join(os.environ["EMB_ROOT"], "manifest.json")))
want = os.environ.get("SCOPE_DS", "").split() or None
tasks, shards = [], {}
for t in m["tasks"]:
    if want is None or t["dataset"] in want:
        tasks.append(str(t["task"]))
        shards[t["dataset"]] = shards.get(t["dataset"], 0) + 1
print(",".join(tasks))
for ds in sorted(shards):
    split = "dev" if ds == "msmarco" else "test"
    print("%s\t%s\t%d" % (ds, split, shards[ds]))
PY
)
TASKS=${SEL[0]}
printf '%s\n' "${SEL[@]:1}" > "$CB_EMB_ROOT/retrieve_targets.tsv"
NRET=$(( ${#SEL[@]} - 1 ))

echo "tag=$TAG scope=$SCOPE encode_tasks=$TASKS retrieve_targets=$NRET"
enc=$(sbatch --parsable --array="$TASKS%${THROTTLE:-100}" \
  --output="$LOGS/cbenc_${TAG}_%A_%a.out" --error="$LOGS/cbenc_${TAG}_%A_%a.err" \
  "$CODE_ROOT/bsc/colbert_encode_array.sbatch")
ret=$(sbatch --parsable --dependency="afterok:$enc" --array="0-$((NRET - 1))%${THROTTLE:-100}" \
  --output="$LOGS/cbret_${TAG}_%A_%a.out" --error="$LOGS/cbret_${TAG}_%A_%a.err" \
  "$CODE_ROOT/bsc/colbert_retrieve_array.sbatch")
echo "encode_job=$enc retrieve_job=$ret"
