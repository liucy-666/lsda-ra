#!/bin/bash
set -x
cd /science/wx/pry/MMDIT || exit 1
P=/science/wx/pry/.venv/bin/python
J=data/FOOD_PILOT/2026_9_15_EXP_7/jobs/aesth
O=data/FOOD_PILOT/2026_9_15_EXP_7/aesth
M=/science/wx/pry/models/stable-diffusion-3.5-large
S=/science/wx/pry/models/sam-vit-base
export CUDA_VISIBLE_DEVICES=0
$P code/lsda/run_census_lsda_v15.py --jobs $J/jobs_v14rect.json --out $O/v14rect --model-dir $M --sam-model-dir $S --variant alpha --feather-px 0 --rect-pad 16 --dilate-px 0
$P code/lsda/run_census_lsda_v15.py --jobs $J/jobs_v15mask.json --out $O/v15mask --model-dir $M --sam-model-dir $S --variant mask --feather-px 12 --dilate-px 24 --rect-pad 16
$P code/lsda/run_census_lsda_v15.py --jobs $J/jobs_v15alpha.json --out $O/v15alpha --model-dir $M --sam-model-dir $S --variant alpha --feather-px 16 --rect-pad 48 --dilate-px 0
echo ALL_DONE
