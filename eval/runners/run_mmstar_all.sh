#!/usr/bin/env bash
# Run MMStar (generate + score_scienceqa) for all 8 VLMs sequentially.
#
# Usage (on GPU server):
#   conda activate venv_neurostrike
#   cd /home/jingliu/workspece/vlm-subspace/vlm-subspace-steering
#   bash eval/runners/run_mmstar_all.sh
#
# Options via env:
#   CUDA_VISIBLE_DEVICES=0   GPU id (default 0)
#   LIMIT=5                  smoke test (first N samples)
#   FORCE=1                  overwrite existing mmstar.jsonl / mmstar_score.csv
#   VLM=qwen25vl             run a single model only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$ROOT/eval"
DATA="$ROOT/data"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"
MANIFEST="$DATA/manifests/mmstar.jsonl"
MAX_TOKENS="${MAX_TOKENS:-192}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV="${VENV:-python3}"
FORCE="${FORCE:-0}"

log() { echo "[$(date +%T)] $*"; }

run_one() {
  local vlm="$1"
  local model_path="$2"
  local out_dir="$OUT_BASE/${vlm}_baseline"
  local gen="$out_dir/mmstar.jsonl"
  local csv="$out_dir/mmstar_score.csv"
  local script="$EVAL/baseline/${vlm}_baseline.py"

  if [[ ! -f "$script" ]]; then
    log "SKIP $vlm (missing $script)"
    return 0
  fi
  if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 1
  fi

  mkdir -p "$out_dir"

  if [[ "$FORCE" == "1" ]]; then
    rm -f "$gen" "$csv"
  fi

  if [[ ! -f "$gen" || ! -s "$gen" ]]; then
    log "GEN $vlm -> $gen"
    local extra=()
    [[ -n "${LIMIT:-}" ]] && extra+=(--limit "$LIMIT")
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VENV" "$script" \
      --model_path "$model_path" \
      --manifest "$MANIFEST" \
      --mode orig \
      --max_new_tokens "$MAX_TOKENS" \
      --out "$gen" \
      "${extra[@]}"
  else
    log "SKIP gen $vlm (exists: $gen)"
  fi

  if [[ ! -f "$csv" || ! -s "$csv" ]]; then
    log "SCORE $vlm -> $csv"
    "$VENV" "$EVAL/score/score_scienceqa.py" \
      --generations "$gen" \
      --manifest "$MANIFEST" \
      --out "$csv"
  else
    log "SKIP score $vlm (exists: $csv)"
  fi
}

# vlm:model_path pairs (same as eval/runners/cmd_shortcut)
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

log "ROOT=$ROOT GPU=$CUDA_VISIBLE_DEVICES MANIFEST=$MANIFEST"

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

for job in "${JOBS[@]}"; do
  run_one "${job%%:*}" "${job#*:}"
done

log "DONE MMStar for all models. Scores:"
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  csv="$OUT_BASE/${name}_baseline/mmstar_score.csv"
  if [[ -f "$csv" ]]; then
    tail -1 "$csv" | awk -v v="$name" '{printf "  %-12s acc=%s (%s/%s)\n", v, $5, $3, $2}'
  else
    echo "  $name MISSING $csv"
  fi
done
