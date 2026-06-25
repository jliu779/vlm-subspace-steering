#!/usr/bin/env bash
# Judge + score + summarize all model subdirectories under a method directory.
#
# Use when outputs are organised as:
#   <METHOD_DIR>/<model_name>/{benchmark}.jsonl   (this script)
# vs. the flat baseline layout:
#   outputs/<model_name>_baseline/{benchmark}.jsonl  (run_baseline_all_models.sh)
#
# Usage:
#   METHOD_DIR=outputs/ecso bash eval/runners/run_judge_score_method_dir.sh
#   METHOD_DIR=outputs/procrustes JUDGE_GPU=1 bash eval/runners/run_judge_score_method_dir.sh
#   METHOD_DIR=outputs/ecso CONTINUE_ON_ERROR=1 bash eval/runners/run_judge_score_method_dir.sh
#   METHOD_DIR=outputs/ecso INCLUDE="gemma3 glm41v" bash eval/runners/run_judge_score_method_dir.sh
#   METHOD_DIR=outputs/ecso EXCLUDE="llava15" bash eval/runners/run_judge_score_method_dir.sh
#
# Env vars forwarded to run_baseline_full.sh:
#   JUDGE_GPU, CUDA_VISIBLE_DEVICES, DUAL_GPU, VENV, JUDGE_CFG, SKIP_JUDGE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_baseline_full.sh"

METHOD_DIR="${METHOD_DIR:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
INCLUDE="${INCLUDE:-}"   # space-separated allowlist; empty = all
EXCLUDE="${EXCLUDE:-}"   # space-separated denylist

log() { echo "[$(date +%T)] $*"; }

if [[ -z "$METHOD_DIR" ]]; then
  echo "ERROR: set METHOD_DIR to the parent directory containing model subdirs." >&2
  echo "  e.g. METHOD_DIR=outputs/ecso bash eval/runners/run_judge_score_method_dir.sh" >&2
  exit 1
fi

# Resolve to absolute path if relative
[[ "$METHOD_DIR" != /* ]] && METHOD_DIR="$ROOT/$METHOD_DIR"

if [[ ! -d "$METHOD_DIR" ]]; then
  echo "ERROR: METHOD_DIR not found: $METHOD_DIR" >&2
  exit 1
fi

# Collect subdirectories (one per model) — avoid mapfile for bash 3.x compat
MODEL_DIRS=()
while IFS= read -r d; do
  MODEL_DIRS+=("$d")
done < <(find "$METHOD_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no subdirectories found in $METHOD_DIR" >&2
  exit 1
fi

log "METHOD_DIR=$METHOD_DIR"
log "Found ${#MODEL_DIRS[@]} model dir(s):"
for d in "${MODEL_DIRS[@]}"; do log "  $(basename "$d")"; done

failed=()

for model_dir in "${MODEL_DIRS[@]}"; do
  model_name="$(basename "$model_dir")"

  # Apply INCLUDE / EXCLUDE filters
  if [[ -n "$INCLUDE" ]]; then
    match=0
    for inc in $INCLUDE; do [[ "$model_name" == "$inc" ]] && match=1 && break; done
    [[ "$match" == "0" ]] && { log "SKIP $model_name (not in INCLUDE)"; continue; }
  fi
  if [[ -n "$EXCLUDE" ]]; then
    skip=0
    for exc in $EXCLUDE; do [[ "$model_name" == "$exc" ]] && skip=1 && break; done
    [[ "$skip" == "1" ]] && { log "SKIP $model_name (in EXCLUDE)"; continue; }
  fi

  log "======== START $model_name ========"
  if OUT_DIR="$model_dir" SKIP_GEN=1 bash "$RUNNER"; then
    log "======== DONE  $model_name ========"
  else
    failed+=("$model_name")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "ERROR: $model_name failed. Set CONTINUE_ON_ERROR=1 to keep going." >&2
      exit 1
    fi
    log "WARN: $model_name failed, continuing..."
  fi
done

log "ALL DONE"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed models: ${failed[*]}" >&2
  exit 1
fi

log "Summaries written:"
for model_dir in "${MODEL_DIRS[@]}"; do
  md="$model_dir/baseline_summary.md"
  model_name="$(basename "$model_dir")"
  [[ -f "$md" ]] && echo "  $md" || echo "  $model_name  MISSING"
done
