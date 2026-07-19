#!/bin/bash
# Progress overview: queue state + encoded shards + finished retrievals.
source "$(dirname "$0")/env.sh"
echo "== queue =="
squeue -u "$USER" -o "%.10i %.9P %.12j %.2t %.10M %R" | head -15
echo "== encode: $(find "$EMB_ROOT" -name 'shard_*.done' 2>/dev/null | wc -l) shards done" \
     "/ $("$VENV/bin/python" -c "import json;print(len(json.load(open('$EMB_ROOT/manifest.json'))['tasks']))" 2>/dev/null || echo '?') =="
echo "== retrieve: $(find "$RESULTS_ROOT" -name metrics.json 2>/dev/null | wc -l) datasets done =="
grep -il "error\|traceback" "$LOGS"/*.err 2>/dev/null | tail -5
