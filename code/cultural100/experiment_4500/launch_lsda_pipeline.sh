#!/usr/bin/env bash
set -euo pipefail

ROOT=/science/wx/pry/AAA_Experiment/cultural100_v1
PY=/science/wx/pry/.venv/bin/python
mkdir -p "$ROOT/logs"

while true; do
  image_count=$(find "$ROOT/base/images" -type f -name '*.png' 2>/dev/null | wc -l)
  sidecar_count=$(find "$ROOT/base/sidecars" -type f -name '*.json' 2>/dev/null | wc -l)
  if [[ "$image_count" -eq 2700 && "$sidecar_count" -eq 2700 ]]; then
    break
  fi
  sleep 60
done

mask_pids=()
for shard in 0 1 2 3 4 5; do
  env CUDA_VISIBLE_DEVICES="$shard" "$PY" "$ROOT/prepare_lsda_masks.py" "$shard" 6 \
    > "$ROOT/logs/masks_shard_${shard}.log" 2>&1 &
  mask_pids+=("$!")
done
for pid in "${mask_pids[@]}"; do wait "$pid"; done

mask_count=$(find "$ROOT/lsda/masks" -type f -name 'mask_summary.json' | wc -l)
if [[ "$mask_count" -ne 900 ]]; then
  echo "mask count mismatch: $mask_count" >&2
  exit 2
fi

for shard in 0 1 2 3 4 5; do
  nohup env CUDA_VISIBLE_DEVICES="$shard" "$PY" "$ROOT/generate_lsda_ra.py" "$shard" 6 \
    > "$ROOT/logs/lsda_shard_${shard}.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/logs/lsda_shard_${shard}.pid"
done

echo "LSDA generation launched"
