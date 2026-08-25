#!/usr/bin/env bash
set -euo pipefail

ROOT=/science/wx/pry/AAA_Experiment/cultural100_v1
PY=/science/wx/pry/.venv/bin/python
mkdir -p "$ROOT/logs"

while true; do
  image_count=$(find "$ROOT/base/images" -type f -name '*.png' 2>/dev/null | wc -l)
  sidecar_count=$(find "$ROOT/base/sidecars" -type f -name '*.json' 2>/dev/null | wc -l)
  if [[ "$image_count" -eq 2700 && "$sidecar_count" -eq 2700 ]]; then break; fi
  sleep 60
done

run_six_and_wait() {
  local script=$1
  local prefix=$2
  local pids=()
  for shard in 0 1 2 3 4 5; do
    env CUDA_VISIBLE_DEVICES="$shard" "$PY" "$ROOT/$script" "$shard" 6 \
      > "$ROOT/logs/${prefix}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

run_six_and_wait prepare_lsda_original_masks.py original_masks
original_mask_count=$(find "$ROOT/lsda_original/masks" -type f -name mask_summary.json | wc -l)
[[ "$original_mask_count" -eq 900 ]] || { echo "original mask count=$original_mask_count" >&2; exit 2; }

run_six_and_wait generate_lsda_original.py original_lsda
original_count=$(find "$ROOT/lsda_original/images" -type f -name '*.png' | wc -l)
[[ "$original_count" -eq 900 ]] || { echo "original LSDA count=$original_count" >&2; exit 3; }

run_six_and_wait prepare_lsda_masks.py ra_masks
ra_mask_count=$(find "$ROOT/lsda/masks" -type f -name mask_summary.json | wc -l)
[[ "$ra_mask_count" -eq 900 ]] || { echo "RA mask count=$ra_mask_count" >&2; exit 4; }

for shard in 0 1 2 3 4 5; do
  nohup env CUDA_VISIBLE_DEVICES="$shard" "$PY" "$ROOT/generate_lsda_ra.py" "$shard" 6 \
    > "$ROOT/logs/ra_lsda_shard_${shard}.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/logs/ra_lsda_shard_${shard}.pid"
done
echo "LSDA-RA generation launched"
