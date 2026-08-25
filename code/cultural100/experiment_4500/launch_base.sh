#!/usr/bin/env bash
set -euo pipefail

ROOT=/science/wx/pry/AAA_Experiment/cultural100_v1
PY=/science/wx/pry/.venv/bin/python
mkdir -p "$ROOT/logs"

for shard in 0 1 2 3 4 5; do
  nohup env CUDA_VISIBLE_DEVICES="$shard" "$PY" "$ROOT/generate_base.py" "$shard" 6 \
    > "$ROOT/logs/base_shard_${shard}.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/logs/base_shard_${shard}.pid"
done

echo "launched 6 base-generation shards"
