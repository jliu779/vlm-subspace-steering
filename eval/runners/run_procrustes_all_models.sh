#!/usr/bin/env bash
# Run full Procrustes-MPC pipeline (all 14 benchmarks) for all 8 VLMs sequentially.
#
# Default: MODE=refp (α=0.2, λ=1, refusal_projected) — paper §5 main config.
# Run all 3 ablation modes: MODES=refp,lambda0,lambda1_full
#
# Usage (on GPU server):
#   conda activate venv_neurostrike
#   cd /home/jingliu/workspece/vlm-subspace/vlm-subspace-steering
#   PR_DIR=/home/jingliu/workspece/vlm-subspace/ProcrustesRotation \
#     bash eval/runners/run_procrustes_all_models.sh
#
# Options via env:
#   VLM=qwen25vl              run one model only
#   MODE=refp                 steering mode (refp | lambda0 | lambda1_full)
#   MODES=refp,lambda0        comma-separated; overrides MODE when set
#   LIMIT=5                   smoke test
#   SKIP_GEN=1 / SKIP_JUDGE=1 forwarded to each run
#   CONTINUE_ON_ERROR=1       keep going if one model fails
#   CUDA_VISIBLE_DEVICES=0    gen GPU (default 0)
#   JUDGE_GPU=1               judge GPU (default 1)
#   DUAL_GPU=1                pipeline gen+judge (default 1)
#   PR_DIR=...                ProcrustesRotation root (auto-detected if sibling)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_procrustes_full.sh"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
MODES="${MODES:-${MODE:-refp}}"

log() { echo "[$(date +%T)] $*"; }

declare -a JOBS=(
  "qwen25vl:/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
  "qwen3vl:/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
  "internvl:/hub/huggingface/models/OpenGVLab/InternVL3_5-8B"
  "internvl3:/hub/huggingface/models/OpenGVLab/InternVL3-8B"
  "llava_next:/hub/huggingface/models/llava-hf/llava-v1.6-mistral-7b-hf"
  "llava15:/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"
  "phi35v:/hub/huggingface/models/microsoft/Phi-3.5-vision-instruct"
  "gemma3:/hub/huggingface/models/google/gemma-3-4b-it"
)

preflight_all_params() {
  local pr="${PR_DIR:-}"
  if [[ -z "$pr" ]]; then
    if [[ -d "$ROOT/ProcrustesRotation" ]]; then
      pr="$ROOT/ProcrustesRotation"
    elif [[ -d "$ROOT/../ProcrustesRotation" ]]; then
      pr="$(cd "$ROOT/../ProcrustesRotation" && pwd)"
    else
      pr="$ROOT/ProcrustesRotation"
    fi
  fi
  log "preflight: checking Procrustes params under $pr/outputs/"
  local missing=0
  local job name
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    if [[ -n "${VLM:-}" && "$name" != "$VLM" ]]; then
      continue
    fi
    local params="$pr/outputs/${name}_procrustes_params_k16.pt"
    if [[ ! -f "$params" ]]; then
      echo "  MISSING $params" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "ERROR: set PR_DIR to your ProcrustesRotation directory" >&2
    exit 1
  fi
  log "preflight: all required .pt params found"
}

run_one() {
  local vlm="$1"
  local model_path="$2"
  local mode="$3"
  log "======== START $vlm MODE=$mode ========"
  MODE="$mode" VLM="$vlm" MODEL_PATH="$model_path" bash "$RUNNER"
  log "======== DONE  $vlm MODE=$mode ========"
}

cd "$ROOT"
log "ROOT=$ROOT"
log "MODES=$MODES"
log "GEN_GPU=${CUDA_VISIBLE_DEVICES:-0} JUDGE_GPU=${JUDGE_GPU:-1} DUAL_GPU=${DUAL_GPU:-1}"
preflight_all_params

IFS=',' read -r -a MODE_LIST <<< "$MODES"

if [[ -n "${VLM:-}" ]]; then
  found=0
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    path="${job#*:}"
    if [[ "$name" == "$VLM" ]]; then
      found=1
      for mode in "${MODE_LIST[@]}"; do
        mode="$(echo "$mode" | xargs)"
        [[ -z "$mode" ]] && continue
        log "===== STEERING MODE: $mode ====="
        run_one "$name" "$path" "$mode"
      done
      break
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "ERROR: unknown VLM='$VLM'" >&2
    exit 1
  fi
  exit 0
fi

for mode in "${MODE_LIST[@]}"; do
  mode="$(echo "$mode" | xargs)"
  if [[ -z "$mode" ]]; then
    continue
  fi
  log "===== STEERING MODE: $mode ====="

  failed=()
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    path="${job#*:}"
    if ! run_one "$name" "$path" "$mode"; then
      failed+=("$name")
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        echo "ERROR: $name (MODE=$mode) failed; set CONTINUE_ON_ERROR=1 to continue" >&2
        exit 1
      fi
      log "WARN: $name failed, continuing..."
    fi
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed models (MODE=$mode): ${failed[*]}" >&2
    exit 1
  fi
done

log "ALL MODELS × MODES FINISHED"
log "Summaries:"
for mode in "${MODE_LIST[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -z "$mode" ]] && continue
  case "$mode" in
    refp) suffix="a02_refp" ;;
    lambda0) suffix="a02_lambda0" ;;
    lambda1_full) suffix="a02_lambda1_full" ;;
    *) suffix="$mode" ;;
  esac
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    md="$ROOT/outputs/${name}_${suffix}/procrustes_summary.md"
    if [[ -f "$md" ]]; then
      echo "  $md"
    else
      echo "  $name/$mode MISSING $md"
    fi
  done
done
