#!/usr/bin/env bash
# Run full baseline pipeline for one VLM:
#   generate -> judge/score -> summary markdown
#
# Default: dual-GPU pipeline (DUAL_GPU=1)
#   GPU 0 (CUDA_VISIBLE_DEVICES): VLM generation
#   GPU 1 (JUDGE_GPU): Llama judge (runs in background while next gen proceeds)
#
# Usage:
#   bash eval/runners/run_baseline_full.sh
#   VLM=qwen25vl MODEL_PATH=/hub/.../Qwen2.5-VL-7B-Instruct bash eval/runners/run_baseline_full.sh
#   VLM=qwen25vl LIMIT=5 bash eval/runners/run_baseline_full.sh
#   SKIP_GEN=1 bash eval/runners/run_baseline_full.sh
#   SKIP_JUDGE=1 bash eval/runners/run_baseline_full.sh
#   DUAL_GPU=0 bash eval/runners/run_baseline_full.sh   # single-GPU sequential
#   VLM=phi4 PHI4_ATTN_IMPLEMENTATION=flash_attention_2 bash eval/runners/run_baseline_full.sh
#   VLM=phi4 PHI4_DISABLE_CACHE=1 bash eval/runners/run_baseline_full.sh  # slow compatibility fallback
#
# Judge + score an arbitrary pre-generated output dir (no VLM baseline script needed):
#   OUT_DIR=outputs/my_method SKIP_GEN=1 bash eval/runners/run_baseline_full.sh
set -euo pipefail

# ===== CONFIG (edit these) =====
VLM="${VLM:-qwen25vl}"                          # qwen25vl | qwen3vl | internvl | internvl3 | llava_next | llava15 | phi35v | gemma3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # VLM generation GPU
JUDGE_GPU="${JUDGE_GPU:-1}"                     # judge GPU (default: second card)
DUAL_GPU="${DUAL_GPU:-1}"                       # 1 = pipeline gen@GPU0 + judge@GPU1
VENV="${VENV:-python3}"
MODEL_PATH="${MODEL_PATH:-}"
JUDGE_CFG="${JUDGE_CFG:-}"
LIMIT="${LIMIT:-}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
PHI4_ATTN_IMPLEMENTATION="${PHI4_ATTN_IMPLEMENTATION:-eager}" # eager | sdpa | flash_attention_2
PHI4_DISABLE_CACHE="${PHI4_DISABLE_CACHE:-0}"
# ==============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$ROOT/eval"
DATA="$ROOT/data"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"
# OUT_DIR can be set directly to support arbitrary method output dirs
# (e.g. OUT_DIR=outputs/my_method SKIP_GEN=1 bash run_baseline_full.sh)
OUT_DIR="${OUT_DIR:-$OUT_BASE/${VLM}_baseline}"
# Normalise to absolute path (relative paths are resolved from ROOT).
[[ "$OUT_DIR" != /* ]] && OUT_DIR="$ROOT/$OUT_DIR"

JUDGE_PIDS=()

if [[ -z "$JUDGE_CFG" ]]; then
  JUDGE_CFG="$EVAL/configs/judge_default.yaml"
fi

if [[ ! -f "$JUDGE_CFG" ]]; then
  echo "ERROR: judge config not found: $JUDGE_CFG" >&2
  echo "Create it or set JUDGE_CFG=/path/to/your/judge.yaml" >&2
  exit 1
fi

# BASELINE_SCRIPT is only needed for generation; skip this check when SKIP_GEN=1
BASELINE_SCRIPT="$EVAL/baseline/${VLM}_baseline.py"
if [[ "$SKIP_GEN" != "1" && ! -f "$BASELINE_SCRIPT" ]]; then
  echo "ERROR: unknown VLM '$VLM' (missing $BASELINE_SCRIPT)" >&2
  echo "Supported: qwen25vl qwen3vl internvl internvl3 llava_next llava15 phi35v gemma3 phi4 glm41v" >&2
  echo "To judge/score an existing output dir, set SKIP_GEN=1 and OUT_DIR=<path>." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

log() { echo "[$(date +%T)] $*"; }

judge_gpu() {
  if [[ "$DUAL_GPU" == "1" ]]; then
    echo "$JUDGE_GPU"
  else
    echo "$CUDA_VISIBLE_DEVICES"
  fi
}

check_judges() {
  # Non-blocking: reap any already-finished judge jobs and fail fast on error.
  local still_running=()
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      # Process already exited — collect its status.
      if ! wait "$pid"; then
        log "ERROR: background judge PID $pid failed"
        # Kill remaining background judges before exiting.
        for p in "${JUDGE_PIDS[@]}"; do
          kill "$p" 2>/dev/null || true
        done
        exit 1
      fi
    else
      still_running+=("$pid")
    fi
  done
  JUDGE_PIDS=("${still_running[@]+"${still_running[@]}"}")
}

wait_judges() {
  if [[ ${#JUDGE_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  log "waiting for ${#JUDGE_PIDS[@]} background judge job(s)..."
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || { log "ERROR: background judge PID $pid failed"; exit 1; }
  done
  JUDGE_PIDS=()
}

_run_gen_on_gpu() {
  local gpu="$1"
  local manifest="$2"
  local out="$3"
  local max_tokens="$4"
  local tmp="${out}.tmp"

  local extra=()
  if [[ -n "$LIMIT" ]]; then
    extra+=(--limit "$LIMIT")
  fi
  local model_args=()
  if [[ -n "$MODEL_PATH" ]]; then
    model_args+=(--model_path "$MODEL_PATH")
  fi
  if [[ "$VLM" == "phi4" ]]; then
    model_args+=(--attn_implementation "$PHI4_ATTN_IMPLEMENTATION")
    if [[ "$PHI4_DISABLE_CACHE" == "1" ]]; then
      model_args+=(--disable_cache)
    fi
  fi
  # Write to tmp file; atomically rename on success so partial output is
  # never mistaken for a complete run on restart.
  ( export CUDA_VISIBLE_DEVICES="$gpu"
    "$VENV" "$BASELINE_SCRIPT" \
      --manifest "$manifest" \
      --mode orig \
      --max_new_tokens "$max_tokens" \
      --out "$tmp" \
      "${model_args[@]}" \
      "${extra[@]}"
  )
  mv "$tmp" "$out"
}

run_gen() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  local out="$OUT_DIR/${stem}.jsonl"

  # Fail fast if any background judge already exited with error.
  check_judges

  if [[ -f "$out" && -s "$out" ]]; then
    log "SKIP gen $stem (exists)"
    return 0
  fi

  if [[ "$DUAL_GPU" == "1" && ${#JUDGE_PIDS[@]} -gt 0 ]]; then
    log "RUN gen $stem on GPU $CUDA_VISIBLE_DEVICES (max_new_tokens=$max_tokens); GPU $(judge_gpu): ${#JUDGE_PIDS[@]} judge job(s) in background"
  else
    log "RUN gen $stem on GPU $CUDA_VISIBLE_DEVICES (max_new_tokens=$max_tokens)"
  fi
  _run_gen_on_gpu "$CUDA_VISIBLE_DEVICES" "$manifest" "$out" "$max_tokens"
}

run_actionable_judge() {
  local stem="$1"
  local bg="${2:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  if [[ ! -f "$gen" ]]; then
    log "SKIP judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP actionable judge $stem (exists)"
    return 0
  fi

  _run_actionable_judge_impl() {
    local _tmp="${judged}.tmp"
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      "$VENV" "$EVAL/judge/judge_actionable_safety.py" \
        --config "$JUDGE_CFG" \
        --judge_style actionable \
        --generations "$gen" \
        --out "$_tmp"
    )
    mv "$_tmp" "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE actionable judge $stem on GPU $gpu (background)"
    _run_actionable_judge_impl &
    JUDGE_PIDS+=("$!")
  else
    log "RUN actionable judge $stem on GPU $gpu"
    _run_actionable_judge_impl
  fi
}

run_context_judge() {
  local stem="$1"
  local manifest="$2"
  local bg="${3:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  if [[ ! -f "$gen" ]]; then
    log "SKIP context judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP context judge $stem (exists)"
    return 0
  fi

  _run_context_judge_impl() {
    local _tmp="${judged}.tmp"
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      "$VENV" "$EVAL/judge/judge_context_aware.py" \
        --config "$JUDGE_CFG" \
        --generations "$gen" \
        --manifest "$manifest" \
        --out "$_tmp"
    )
    mv "$_tmp" "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE context judge $stem on GPU $gpu (background)"
    _run_context_judge_impl &
    JUDGE_PIDS+=("$!")
  else
    log "RUN context judge $stem on GPU $gpu"
    _run_context_judge_impl
  fi
}

run_over_refusal_judge() {
  local stem="$1"
  local bg="${2:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  if [[ ! -f "$gen" ]]; then
    log "SKIP over-refusal judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP over-refusal judge $stem (exists)"
    return 0
  fi

  _run_over_refusal_judge_impl() {
    local _tmp="${judged}.tmp"
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      "$VENV" "$EVAL/judge/judge_over_refusal.py" \
        --config "$JUDGE_CFG" \
        --generations "$gen" \
        --out "$_tmp"
    )
    mv "$_tmp" "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE over-refusal judge $stem on GPU $gpu (background)"
    _run_over_refusal_judge_impl &
    JUDGE_PIDS+=("$!")
  else
    log "RUN over-refusal judge $stem on GPU $gpu"
    _run_over_refusal_judge_impl
  fi
}

schedule_gpu_judges_after_gen() {
  local stem="$1"
  local manifest="${2:-}"
  local bg=0
  if [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]]; then
    bg=1
  fi

  case "$stem" in
    vlsafe_examine_eval|spa_vl_test_530|mmsb_vision_risk_sdtypo|mm_safetybench_300)
      run_actionable_judge "$stem" "$bg"
      ;;
    siuo_167)
      run_context_judge "$stem" "$manifest" "$bg"
      ;;
    mssbench_unsafe_full)
      run_context_judge "$stem" "$manifest" "$bg"
      ;;
    benign_multimodal_n60|mossbench|xstest_safe)
      run_over_refusal_judge "$stem" "$bg"
      ;;
  esac
}

run_gen_and_maybe_judge() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  run_gen "$stem" "$manifest" "$max_tokens"
  schedule_gpu_judges_after_gen "$stem" "$manifest"
}

run_mcq_score() {
  local stem="$1"
  local manifest="$2"
  local csv_name="${3:-${stem}_score.csv}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/${csv_name}"
  if [[ ! -f "$gen" ]]; then
    log "SKIP mcq score $stem (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP mcq score $stem (exists)"
    return 0
  fi
  log "RUN mcq score $stem (CPU)"
  "$VENV" "$EVAL/score/score_scienceqa.py" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$csv"
}

run_mathvista_score() {
  local stem="mathvista"
  local manifest="$DATA/manifests/mathvista.jsonl"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/${stem}_score.csv"
  if [[ ! -f "$gen" ]]; then
    log "SKIP mathvista score (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP mathvista score (exists)"
    return 0
  fi
  log "RUN mathvista score (CPU)"
  "$VENV" "$EVAL/score/score_mathvista.py" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$csv"
}

run_colorbench_score() {
  local stem="colorbench"
  local manifest="$DATA/manifests/colorbench.jsonl"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/${stem}_score.csv"
  if [[ ! -f "$gen" ]]; then
    log "SKIP colorbench score (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP colorbench score (exists)"
    return 0
  fi
  log "RUN colorbench score (CPU)"
  "$VENV" "$EVAL/score/score_mathvista.py" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$csv"
}

run_cpu_scores() {
  run_mcq_score "scienceqa_imgval_full" "$DATA/manifests/scienceqa_imgval_full.jsonl" "sciqa_full_score.csv"
  run_mcq_score "mmstar" "$DATA/manifests/mmstar.jsonl" "mmstar_score.csv"
  run_mcq_score "mme_realworld" "$DATA/manifests/mme_realworld.jsonl" "mme_realworld_score.csv"
  run_mathvista_score
  run_colorbench_score
}

run_all_gpu_judges_sequential() {
  run_actionable_judge "vlsafe_examine_eval"
  run_actionable_judge "spa_vl_test_530"
  run_actionable_judge "mmsb_vision_risk_sdtypo"
  run_actionable_judge "mm_safetybench_300"
  run_context_judge "siuo_167" "$DATA/manifests/siuo_167.jsonl"
  run_context_judge "mssbench_unsafe_full" "$DATA/manifests/mssbench_unsafe_full.jsonl"
  run_over_refusal_judge "benign_multimodal_n60"
  run_over_refusal_judge "mossbench"
  run_over_refusal_judge "xstest_safe"
}

log "ROOT=$ROOT"
log "VLM=$VLM OUT_DIR=$OUT_DIR"
log "GEN_GPU=$CUDA_VISIBLE_DEVICES JUDGE_GPU=$(judge_gpu) DUAL_GPU=$DUAL_GPU"
log "VENV=$VENV JUDGE_CFG=$JUDGE_CFG"

if [[ "$SKIP_GEN" != "1" ]]; then
  if [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]]; then
    log "pipeline mode: gen on GPU $CUDA_VISIBLE_DEVICES, judge on GPU $(judge_gpu)"
    run_gen_and_maybe_judge "vlsafe_examine_eval"      "$DATA/manifests/vlsafe_examine_eval.jsonl"      256
    run_gen_and_maybe_judge "spa_vl_test_530"          "$DATA/manifests/spa_vl_test_530.jsonl"          192
    run_gen_and_maybe_judge "mmsb_vision_risk_sdtypo"  "$DATA/manifests/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen_and_maybe_judge "mm_safetybench_300"       "$DATA/manifests/mm_safetybench_300.jsonl"       192
    run_gen_and_maybe_judge "siuo_167"                 "$DATA/manifests/siuo_167.jsonl"                 192
    run_gen_and_maybe_judge "mssbench_unsafe_full"     "$DATA/manifests/mssbench_unsafe_full.jsonl"     192
    # Small OR benchmarks first so GPU1 can judge while long utility gens run on GPU0.
    run_gen_and_maybe_judge "mossbench"                "$DATA/manifests/mossbench.jsonl"                192
    run_gen_and_maybe_judge "xstest_safe"              "$DATA/manifests/xstest_safe.jsonl"              192
    run_gen_and_maybe_judge "benign_multimodal_n60"    "$DATA/manifests/benign_multimodal_n60.jsonl"    192
    run_gen "scienceqa_imgval_full"    "$DATA/manifests/scienceqa_imgval_full.jsonl"    192
    run_gen "mmstar"                   "$DATA/manifests/mmstar.jsonl"                   192
    run_gen "mme_realworld"            "$DATA/manifests/mme_realworld.jsonl"            192
    run_gen "mathvista"                "$DATA/manifests/mathvista.jsonl"                256
    run_gen "colorbench"               "$DATA/manifests/colorbench.jsonl"               192
    wait_judges
  else
    run_gen "vlsafe_examine_eval"      "$DATA/manifests/vlsafe_examine_eval.jsonl"      256
    run_gen "spa_vl_test_530"          "$DATA/manifests/spa_vl_test_530.jsonl"          192
    run_gen "mmsb_vision_risk_sdtypo"  "$DATA/manifests/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen "mm_safetybench_300"       "$DATA/manifests/mm_safetybench_300.jsonl"       192
    run_gen "siuo_167"                 "$DATA/manifests/siuo_167.jsonl"                 192
    run_gen "mssbench_unsafe_full"     "$DATA/manifests/mssbench_unsafe_full.jsonl"     192
    run_gen "mossbench"                "$DATA/manifests/mossbench.jsonl"                192
    run_gen "xstest_safe"              "$DATA/manifests/xstest_safe.jsonl"              192
    run_gen "benign_multimodal_n60"    "$DATA/manifests/benign_multimodal_n60.jsonl"    192
    run_gen "scienceqa_imgval_full"    "$DATA/manifests/scienceqa_imgval_full.jsonl"    192
    run_gen "mmstar"                   "$DATA/manifests/mmstar.jsonl"                   192
    run_gen "mme_realworld"            "$DATA/manifests/mme_realworld.jsonl"            192
    run_gen "mathvista"                "$DATA/manifests/mathvista.jsonl"                256
    run_gen "colorbench"               "$DATA/manifests/colorbench.jsonl"               192
  fi
else
  log "SKIP_GEN=1, generation phase skipped"
fi

if [[ "$SKIP_JUDGE" != "1" ]]; then
  if [[ "$SKIP_GEN" == "1" || "$DUAL_GPU" != "1" ]]; then
    run_all_gpu_judges_sequential
  fi
  run_cpu_scores

  SUMMARY_MD="$OUT_DIR/baseline_summary.md"
  METHOD_TAG="$(basename "$OUT_DIR")"
  "$VENV" "$EVAL/aggregate/summarize_baseline_metrics.py" \
    --out_dir "$OUT_DIR" \
    --method "$METHOD_TAG" \
    --out_md "$SUMMARY_MD"
  log "Summary written: $SUMMARY_MD"
else
  log "SKIP_JUDGE=1, judging/scoring phase skipped"
fi

log "DONE baseline pipeline for $VLM"
log "Outputs: $OUT_DIR"
