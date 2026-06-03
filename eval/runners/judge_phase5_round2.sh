#!/usr/bin/env bash
# Phase 5 round 2 judge: wait for round 2 generations (PIDs 3873431, 3873432)
# to finish, then judge vlsafe1110 + mmsb_sdtypo on freed GPUs 4/5 in parallel.
set -u

VENV=/home/kedong/repos/VLM-subspace-steering/CMRM/.venv/bin/python
CMRM=/home/kedong/repos/VLM-subspace-steering/CMRM
PR=/home/kedong/repos/VLM-subspace-steering/ProcrustesRotation
BASE=$PR/outputs/phase5_smoke_2026-05-27
CFG=$CMRM/configs/llava7b_vlsafe.yaml

cd $PR

# Wait for round 2 to finish (poll once per minute)
while kill -0 3873431 2>/dev/null || kill -0 3873432 2>/dev/null; do
  sleep 30
done
echo "[$(date +%T)] round-2 generations exited; starting judges"

# Two lanes, each gets one VLM × {vlsafe1110, mmsb_sdtypo}
judge_one() {
  local gpu=$1; local vlm=$2
  local d=$BASE/${vlm}_a02_refp
  for name in vlsafe1110 mmsb_sdtypo; do
    local gen=$d/${name}.jsonl
    local out=$d/${name}.judged.jsonl
    if [[ ! -f "$gen" || ! -s "$gen" ]]; then
      echo "[$(date +%T)] gpu=$gpu SKIP missing $gen"; continue
    fi
    if [[ -f "$out" && -s "$out" ]]; then
      echo "[$(date +%T)] gpu=$gpu SKIP existing $out"; continue
    fi
    echo "[$(date +%T)] gpu=$gpu JUDGE ${vlm}_a02/${name}"
    CUDA_VISIBLE_DEVICES=$gpu $VENV $CMRM/scripts/07_judge_outputs.py \
      --config $CFG --generations "$gen" --judge_style actionable --out "$out"
    rc=$?
    echo "[$(date +%T)] gpu=$gpu DONE ${vlm}_a02/${name} rc=$rc"
    if [[ $rc -ne 0 ]]; then echo "abort lane $gpu" >&2; return $rc; fi
  done
}

judge_one 4 qwen25vl > $PR/logs/phase5_smoke/round2_judge_gpu4.log 2>&1 &
P1=$!
judge_one 5 internvl25 > $PR/logs/phase5_smoke/round2_judge_gpu5.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[$(date +%T)] round-2 judging complete"
