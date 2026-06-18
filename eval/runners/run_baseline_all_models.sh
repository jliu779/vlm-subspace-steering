#!/usr/bin/env bash
# Run full baseline (all benchmarks) for 8 VLMs sequentially.
#
# Usage (on GPU server):
#   conda activate venv_neurostrike
#   cd /home/jingliu/workspece/vlm-subspace/vlm-subspace-steering
#   bash eval/runners/run_baseline_all_models.sh
#
# Options via env:
#   VLM=qwen25vl              run one model only
#   LIMIT=5                   smoke test (passed to run_baseline_full.sh)
#   SKIP_GEN=1 / SKIP_JUDGE=1  forwarded to each model run
#   CONTINUE_ON_ERROR=1       keep going if one model fails
#   CUDA_VISIBLE_DEVICES=0    gen GPU (default 0)
#   JUDGE_GPU=1               judge GPU (default 1)
#   DUAL_GPU=1                pipeline gen+judge (default 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_baseline_full.sh"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

log() { echo "[$(date +%T)] $*"; }

declare -a JOBS=(
  "llava15:/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"
  "qwen25vl:/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
  "internvl3:/hub/huggingface/models/OpenGVLab/InternVL3-8B"
  "qwen3vl:/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
  "internvl:/hub/huggingface/models/OpenGVLab/InternVL3_5-8B"
  "gemma3:/hub/huggingface/models/google/gemma-3-4b-it"
  "phi4:/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct"
  "glm41v:/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking"
)

run_one() {
  local vlm="$1"
  local model_path="$2"
  log "======== START $vlm ========"
  VLM="$vlm" MODEL_PATH="$model_path" bash "$RUNNER"
  log "======== DONE  $vlm ========"
}

cd "$ROOT"
log "ROOT=$ROOT"
log "GEN_GPU=${CUDA_VISIBLE_DEVICES:-0} JUDGE_GPU=${JUDGE_GPU:-1} DUAL_GPU=${DUAL_GPU:-1}"

if [[ -n "${VLM:-}" ]]; then
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    path="${job#*:}"
    if [[ "$name" == "$VLM" ]]; then
      run_one "$name" "$path"
      exit 0
    fi
  done
  echo "ERROR: unknown VLM='$VLM'" >&2
  exit 1
fi

failed=()
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  path="${job#*:}"
  if ! run_one "$name" "$path"; then
    failed+=("$name")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "ERROR: $name failed; set CONTINUE_ON_ERROR=1 to run remaining models" >&2
      exit 1
    fi
    log "WARN: $name failed, continuing..."
  fi
done

log "ALL MODELS FINISHED"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed models: ${failed[*]}" >&2
  exit 1
fi

log "Summaries:"
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  md="$ROOT/outputs/${name}_baseline/baseline_summary.md"
  if [[ -f "$md" ]]; then
    echo "  $md"
  else
    echo "  $name MISSING $md"
  fi
done
