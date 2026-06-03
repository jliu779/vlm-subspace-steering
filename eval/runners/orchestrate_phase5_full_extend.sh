#!/usr/bin/env bash
# Two-lane orchestrator: wait for the current phase5_extend processes (qwen3vl
# on GPU 4, internvl3 on GPU 5) to finish, then run the full-cell extension
# (B, C/mmsb, D/siuo, D/mssb) for the corresponding pair of VLMs on each lane.
#
# Lane 4: phase5_extend qwen3vl  →  qwen25vl full-extend  →  qwen3vl full-extend
# Lane 5: phase5_extend internvl3 →  internvl25 full-extend →  internvl3 full-extend
set -u

PR=/home/kedong/repos/VLM-subspace-steering/ProcrustesRotation
SCRIPT=$PR/scripts/run_phase5_full_extend.sh
LOGDIR=$PR/logs/phase5_full_extend
mkdir -p "$LOGDIR"

run_lane() {
  local gpu=$1; local current_pid=$2; local vlm_now=$3; local vlm_after=$4
  while kill -0 "$current_pid" 2>/dev/null; do sleep 30; done
  echo "[$(date +%T)] lane gpu=$gpu phase5_extend done (vlm=$vlm_now); starting full-extend $vlm_now"
  VLM=$vlm_now GPU=$gpu bash "$SCRIPT" > "$LOGDIR/${vlm_now}.log" 2>&1
  echo "[$(date +%T)] lane gpu=$gpu starting full-extend $vlm_after"
  VLM=$vlm_after GPU=$gpu bash "$SCRIPT" > "$LOGDIR/${vlm_after}.log" 2>&1
  echo "[$(date +%T)] lane gpu=$gpu DONE"
}

# Current PIDs: 85979 (qwen3vl on GPU 4), 91567 (internvl3 on GPU 5)
run_lane 4 85979 qwen25vl qwen3vl  > "$LOGDIR/lane_gpu4.log" 2>&1 &
P4=$!
run_lane 5 91567 internvl25 internvl3 > "$LOGDIR/lane_gpu5.log" 2>&1 &
P5=$!
wait $P4 $P5
echo "[$(date +%T)] orchestrator complete"
