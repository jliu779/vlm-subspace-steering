#!/usr/bin/env bash
# Full Procrustes-MPC pipeline for one VLM (all 14 benchmarks):
#   generate (with steering hooks) -> judge/score -> summary markdown
#
# Default steering: refp α=0.2, λ=1 (paper §5 picked config).
# Other modes: MODE=lambda0 | lambda1_full
#
# Dual-GPU pipeline (DUAL_GPU=1, default):
#   GPU 0 (CUDA_VISIBLE_DEVICES): VLM generation
#   GPU 1 (JUDGE_GPU): Llama judge (background while next gen runs)
#
# Usage:
#   bash eval/runners/run_procrustes_full.sh
#   VLM=qwen25vl MODEL_PATH=/hub/.../Qwen2.5-VL-7B-Instruct bash eval/runners/run_procrustes_full.sh
#   MODE=lambda0 VLM=qwen25vl bash eval/runners/run_procrustes_full.sh
#   VLM=qwen25vl LIMIT=5 bash eval/runners/run_procrustes_full.sh
#   SKIP_GEN=1 / SKIP_JUDGE=1  — same as baseline runner
#   PR_DIR=/path/to/ProcrustesRotation bash eval/runners/run_procrustes_full.sh
#   GEN_SHARD=auto bash ...  # split large benchmarks across GPU0+GPU1 (default)
#   GEN_SHARD=0 bash ...     # disable dual-GPU sharded generation
set -euo pipefail

# ===== CONFIG (edit these) =====
VLM="${VLM:-qwen25vl}"
MODE="${MODE:-refp}"                            # refp | lambda0 | lambda1_full
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
DUAL_GPU="${DUAL_GPU:-1}"
GEN_SHARD="${GEN_SHARD:-auto}"                    # auto|2 = shard large manifests on both GPUs; 0 = off
SHARD_MIN_LINES="${SHARD_MIN_LINES:-500}"         # auto-shard when manifest has >= this many lines
VENV="${VENV:-python3}"
MODEL_PATH="${MODEL_PATH:-}"
JUDGE_CFG="${JUDGE_CFG:-}"
PR_DIR="${PR_DIR:-}"
LIMIT="${LIMIT:-}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
# ==============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$ROOT/eval"
DATA="$ROOT/data"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"

if [[ -z "$PR_DIR" ]]; then
  if [[ -d "$ROOT/ProcrustesRotation" ]]; then
    PR_DIR="$ROOT/ProcrustesRotation"
  elif [[ -d "$ROOT/../ProcrustesRotation" ]]; then
    PR_DIR="$(cd "$ROOT/../ProcrustesRotation" && pwd)"
  else
    PR_DIR="$ROOT/ProcrustesRotation"
  fi
fi

JUDGE_PIDS=()

if [[ -z "$JUDGE_CFG" ]]; then
  JUDGE_CFG="$EVAL/configs/judge_default.yaml"
fi

log() { echo "[$(date +%T)] $*"; }

# shellcheck source=lib_gen_shard.sh
source "$SCRIPT_DIR/lib_gen_shard.sh"

resolve_steering_mode() {
  case "$MODE" in
    refp)
      ALPHA="0.2"
      LAMBDA_MEAN="1"
      MEAN_SHIFT_MODE="refusal_projected"
      OUT_SUFFIX="a02_refp"
      METHOD_LABEL="procrustes_refp_a0.2"
      ;;
    lambda0)
      ALPHA="0.2"
      LAMBDA_MEAN="0"
      MEAN_SHIFT_MODE="full"
      OUT_SUFFIX="a02_lambda0"
      METHOD_LABEL="procrustes_lambda0_a0.2"
      ;;
    lambda1_full)
      ALPHA="0.2"
      LAMBDA_MEAN="1"
      MEAN_SHIFT_MODE="full"
      OUT_SUFFIX="a02_lambda1_full"
      METHOD_LABEL="procrustes_lambda1_full_a0.2"
      ;;
    *)
      echo "ERROR: unknown MODE='$MODE' (use refp | lambda0 | lambda1_full)" >&2
      exit 1
      ;;
  esac
}

resolve_vlm_assets() {
  local refusal_stem="$VLM"
  case "$VLM" in
    internvl) refusal_stem="internvl" ;;
  esac

  GENERATE_SCRIPT="$EVAL/generate/${VLM}_procrustes_generate.py"
  PROCRUSTES_PARAMS="$PR_DIR/outputs/${VLM}_procrustes_params_k16.pt"
  REFUSAL_DIR="$PR_DIR/outputs/refusal_dir_${refusal_stem}.pt"
  OUT_DIR="$OUT_BASE/${VLM}_${OUT_SUFFIX}"
  METHOD_TAG="${VLM}_${OUT_SUFFIX}"
}

preflight_check() {
  if [[ ! -f "$JUDGE_CFG" ]]; then
    echo "ERROR: judge config not found: $JUDGE_CFG" >&2
    exit 1
  fi
  if [[ ! -f "$GENERATE_SCRIPT" ]]; then
    echo "ERROR: unknown VLM '$VLM' (missing $GENERATE_SCRIPT)" >&2
    echo "Supported: qwen25vl qwen3vl internvl internvl3 llava_next llava15 phi35v gemma3" >&2
    exit 1
  fi
  if [[ ! -f "$PROCRUSTES_PARAMS" ]]; then
    echo "ERROR: Procrustes params not found: $PROCRUSTES_PARAMS" >&2
    echo "Set PR_DIR to the directory containing ProcrustesRotation/outputs/" >&2
    exit 1
  fi
  if [[ "$MEAN_SHIFT_MODE" == "refusal_projected" && ! -f "$REFUSAL_DIR" ]]; then
    echo "ERROR: refusal direction not found: $REFUSAL_DIR (required for MODE=refp)" >&2
    exit 1
  fi
}

resolve_steering_mode
resolve_vlm_assets
preflight_check
mkdir -p "$OUT_DIR"

judge_gpu() {
  if [[ "$DUAL_GPU" == "1" ]]; then
    echo "$JUDGE_GPU"
  else
    echo "$CUDA_VISIBLE_DEVICES"
  fi
}

wait_judges() {
  if [[ ${#JUDGE_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  log "waiting for ${#JUDGE_PIDS[@]} background judge job(s)..."
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || exit 1
  done
  JUDGE_PIDS=()
}

_run_gen_on_gpu() {
  local gpu="$1"
  local manifest="$2"
  local out="$3"
  local max_tokens="$4"

  local extra=()
  if [[ -n "$LIMIT" ]]; then
    extra+=(--limit "$LIMIT")
  fi

  local steer_args=(
    --manifest "$manifest"
    --params "$PROCRUSTES_PARAMS"
    --alpha "$ALPHA"
    --lambda_mean "$LAMBDA_MEAN"
    --mean_shift_mode "$MEAN_SHIFT_MODE"
    --hook_scope prefill_only
    --mode orig
    --max_new_tokens "$max_tokens"
    --method_tag "$METHOD_TAG"
    --out "$out"
  )
  if [[ "$MEAN_SHIFT_MODE" == "refusal_projected" ]]; then
    steer_args+=(--refusal_dir "$REFUSAL_DIR")
  fi

  local model_args=()
  if [[ -n "$MODEL_PATH" ]]; then
    model_args+=(--model_path "$MODEL_PATH")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$GENERATE_SCRIPT" \
    "${steer_args[@]}" \
    "${model_args[@]}" \
    "${extra[@]}"
}

run_gen_sharded() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  local out="$OUT_DIR/${stem}.jsonl"
  local shard_dir="$OUT_DIR/.shards"
  local m0="$shard_dir/${stem}.shard0.jsonl"
  local m1="$shard_dir/${stem}.shard1.jsonl"
  local o0="$OUT_DIR/${stem}.part0.jsonl"
  local o1="$OUT_DIR/${stem}.part1.jsonl"

  wait_judges
  mkdir -p "$shard_dir"
  shard_split_manifest_halves "$manifest" "$m0" "$m1"
  local n0 n1
  n0="$(shard_count_lines "$m0")"
  n1="$(shard_count_lines "$m1")"
  log "RUN gen $stem SHARDED (MODE=$MODE) on GPU $CUDA_VISIBLE_DEVICES ($n0) + GPU $JUDGE_GPU ($n1)"

  local pid0="" pid1=""
  if [[ ! -f "$o0" || ! -s "$o0" ]]; then
    _run_gen_on_gpu "$CUDA_VISIBLE_DEVICES" "$m0" "$o0" "$max_tokens" &
    pid0=$!
  else
    log "SKIP shard0 gen $stem (exists)"
  fi
  if [[ ! -f "$o1" || ! -s "$o1" ]]; then
    if [[ "$n1" -gt 0 ]]; then
      _run_gen_on_gpu "$JUDGE_GPU" "$m1" "$o1" "$max_tokens" &
      pid1=$!
    else
      : >"$o1"
      log "SKIP shard1 gen $stem (empty shard)"
    fi
  else
    log "SKIP shard1 gen $stem (exists)"
  fi
  if [[ -n "$pid0" ]]; then
    wait "$pid0"
  fi
  if [[ -n "$pid1" ]]; then
    wait "$pid1"
  fi
  shard_merge_jsonl "$out" "$o0" "$o1"
  rm -f "$o0" "$o1"
  log "MERGED sharded gen $stem -> $out"
}

run_gen() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  local out="$OUT_DIR/${stem}.jsonl"

  if [[ -f "$out" && -s "$out" ]]; then
    log "SKIP gen $stem (exists)"
    return 0
  fi

  if should_gen_shard "$manifest"; then
    run_gen_sharded "$stem" "$manifest" "$max_tokens"
    return 0
  fi

  if [[ "$DUAL_GPU" == "1" && ${#JUDGE_PIDS[@]} -gt 0 ]]; then
    log "RUN gen $stem on GPU $CUDA_VISIBLE_DEVICES (Procrustes MODE=$MODE α=$ALPHA λ=$LAMBDA_MEAN); GPU $(judge_gpu): ${#JUDGE_PIDS[@]} judge job(s) in background"
  else
    log "RUN gen $stem on GPU $CUDA_VISIBLE_DEVICES (Procrustes MODE=$MODE α=$ALPHA λ=$LAMBDA_MEAN)"
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
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$EVAL/judge/judge_actionable_safety.py" \
      --config "$JUDGE_CFG" \
      --judge_style actionable \
      --generations "$gen" \
      --out "$judged"
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
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$EVAL/judge/judge_context_aware.py" \
      --config "$JUDGE_CFG" \
      --generations "$gen" \
      --manifest "$manifest" \
      --out "$judged"
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
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$EVAL/judge/judge_over_refusal.py" \
      --config "$JUDGE_CFG" \
      --generations "$gen" \
      --out "$judged"
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
    siuo_167|mssbench_unsafe_full)
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
log "VLM=$VLM MODE=$MODE OUT_DIR=$OUT_DIR"
log "PR_DIR=$PR_DIR"
log "PARAMS=$PROCRUSTES_PARAMS"
if [[ "$MEAN_SHIFT_MODE" == "refusal_projected" ]]; then
  log "REFUSAL_DIR=$REFUSAL_DIR"
fi
log "GEN_GPU=$CUDA_VISIBLE_DEVICES JUDGE_GPU=$(judge_gpu) DUAL_GPU=$DUAL_GPU GEN_SHARD=$GEN_SHARD"
log "VENV=$VENV JUDGE_CFG=$JUDGE_CFG"

if [[ "$SKIP_GEN" != "1" ]]; then
  if [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]]; then
    log "pipeline mode: Procrustes gen on GPU $CUDA_VISIBLE_DEVICES, judge on GPU $(judge_gpu)"
    run_gen_and_maybe_judge "vlsafe_examine_eval"      "$DATA/manifests/vlsafe_examine_eval.jsonl"      256
    run_gen_and_maybe_judge "spa_vl_test_530"          "$DATA/manifests/spa_vl_test_530.jsonl"          192
    run_gen_and_maybe_judge "mmsb_vision_risk_sdtypo"  "$DATA/manifests/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen_and_maybe_judge "mm_safetybench_300"       "$DATA/manifests/mm_safetybench_300.jsonl"       192
    run_gen_and_maybe_judge "siuo_167"                 "$DATA/manifests/siuo_167.jsonl"                 192
    run_gen_and_maybe_judge "mssbench_unsafe_full"     "$DATA/manifests/mssbench_unsafe_full.jsonl"     192
    # Small OR benchmarks first so GPU1 judges while long utility gens run on GPU0.
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

  SUMMARY_MD="$OUT_DIR/procrustes_summary.md"
  "$VENV" "$EVAL/aggregate/summarize_baseline_metrics.py" \
    --out_dir "$OUT_DIR" \
    --method "$METHOD_LABEL" \
    --out_md "$SUMMARY_MD"
  log "Summary written: $SUMMARY_MD"
else
  log "SKIP_JUDGE=1, judging/scoring phase skipped"
fi

log "DONE Procrustes pipeline for $VLM (MODE=$MODE)"
log "Outputs: $OUT_DIR"
