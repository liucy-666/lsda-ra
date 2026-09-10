#!/bin/bash
# Launch pilot counterfactual generation on 2 GPUs (shard 0 -> GPU3, shard 1 -> GPU5)
set -u
BASE=/science/wx/pry/MMDIT/experiment/2026_9_1_EXP_1
cd /science/wx/pry/MMDIT/code/agentic_lsda
COMMON="--manifest $BASE/manifests/counterfactual_pilot.json
  --model-dir /science/wx/pry/models/stable-diffusion-3.5-large
  --native-images-root $BASE/native_images
  --mask-npz-root $BASE/masks
  --image-output-dir $BASE/images
  --record-output-dir $BASE/records"

CUDA_VISIBLE_DEVICES=3 nohup /science/wx/pry/.venv/bin/python collect_counterfactuals.py $COMMON --shard 0 --nshards 2 > $BASE/logs/pilot_shard0.log 2>&1 &
echo "shard0 pid $!"
CUDA_VISIBLE_DEVICES=5 nohup /science/wx/pry/.venv/bin/python collect_counterfactuals.py $COMMON --shard 1 --nshards 2 > $BASE/logs/pilot_shard1.log 2>&1 &
echo "shard1 pid $!"
