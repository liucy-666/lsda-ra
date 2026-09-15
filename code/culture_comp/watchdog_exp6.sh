#!/bin/bash
# EXP_6 watchdog v2: sequence LL (GPU0) then LSDA-Long (GPU0) so they never share a contended GPU.
ROOT=/science/wx/pry/MMDIT
EXP=$ROOT/experiments/2026_9_15_EXP_6_KNOW_INJECT
OUTLL=$ROOT/data/KNOW_INJECT/2026_9_15_EXP_6/LL
OUTLS=$ROOT/data/KNOW_INJECT/2026_9_15_EXP_6/LSDA_Long
PY=/science/wx/pry/.venv/bin/python
MD=/science/wx/pry/models
LOG=$EXP/watchdog.log
while true; do
  nll=$(ls $OUTLL/*.png 2>/dev/null | wc -l)
  nls=$(find $OUTLS -name lsda.png 2>/dev/null | wc -l)
  if [ "$nll" -lt 300 ]; then
    if ! pgrep -f "gen_ll.py" >/dev/null; then
      echo "$(date '+%F %T') restart LL ($nll/300)" >> $LOG
      cd $ROOT && setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY code/culture_comp/gen_ll.py --jobs $EXP/jobs/ll_jobs.json --out $OUTLL \
        --model-dir $MD/stable-diffusion-3.5-large >> $EXP/ll.log 2>&1 < /dev/null &
    fi
  else
    if [ "$nls" -lt 250 ] && ! pgrep -f "run_census_lsda.py" >/dev/null; then
      echo "$(date '+%F %T') start LSDA-Long on GPU0 ($nls/250)" >> $LOG
      sleep 20
      cd $ROOT && setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY code/lsda/run_census_lsda.py --jobs $EXP/jobs/lsdalong_jobs.json \
        --images-dir $ROOT/experiments/2026_9_12_EXP_3_KA_ME/census/images --out $OUTLS \
        --model-dir $MD/stable-diffusion-3.5-large --sam-model-dir $MD/sam-vit-base --rect-pad 16 \
        >> $EXP/lsdalong.log 2>&1 < /dev/null &
    fi
  fi
  echo "$(date '+%F %T') LL=$nll/300 LSDA=$nls/250" >> $LOG
  sleep 180
done
