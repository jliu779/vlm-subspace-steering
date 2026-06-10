#!/usr/bin/env bash
# Run full baseline pipeline for one VLM:
#   generate (8 manifests) -> judge/score -> summary markdown
#
# Usage:
#   # 1) Edit the CONFIG section below, then:
#   bash eval/runners/run_baseline_full.sh
#
#   # 2) Or override via env:
#   VLM=qwen25vl CUDA_VISIBLE_DEVICES=0 LIMIT=5 bash eval/runners/run_baseline_full.sh
#
#   # 3) Re-run only judging (skip generation):
#   SKIP_GEN=1 bash eval/runners/run_baseline_full.sh
#
#   # 4) Re-run only generation (skip judging):
#   SKIP_JUDGE=1 bash eval/runners/run_baseline_full.sh
set -euo pipefail

# ===== CONFIG (edit these) =====
VLM="${VLM:-qwen25vl}"                          # qwen25vl | qwen3vl | internvl | internvl3 | llava_next | llava15 | phi35v | gemma3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV="${VENV:-python3}"                          # e.g. /path/to/.venv/bin/python
MODEL_PATH="${MODEL_PATH:-}"                      # optional override for --model_path
JUDGE_CFG="${JUDGE_CFG:-}"                        # optional override; default: eval/configs/judge_default.yaml
LIMIT="${LIMIT:-}"                                # optional smoke test, e.g. LIMIT=5
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
# ==============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$ROOT/eval"
DATA="$ROOT/data"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"
OUT_DIR="$OUT_BASE/${VLM}_baseline"

if [[ -z "$JUDGE_CFG" ]]; then
  JUDGE_CFG="$EVAL/configs/judge_default.yaml"
fi

if [[ ! -f "$JUDGE_CFG" ]]; then
  echo "ERROR: judge config not found: $JUDGE_CFG" >&2
  echo "Create it or set JUDGE_CFG=/path/to/your/judge.yaml" >&2
  exit 1
fi

BASELINE_SCRIPT="$EVAL/baseline/${VLM}_baseline.py"
if [[ ! -f "$BASELINE_SCRIPT" ]]; then
  echo "ERROR: unknown VLM '$VLM' (missing $BASELINE_SCRIPT)" >&2
  echo "Supported: qwen25vl qwen3vl internvl internvl3 llava_next llava15 phi35v gemma3" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

log() { echo "[$(date +%T)] $*"; }

run_gen() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  local out="$OUT_DIR/${stem}.jsonl"

  if [[ -f "$out" && -s "$out" ]]; then
    log "SKIP gen $stem (exists)"
    return 0
  fi

  local extra=()
  if [[ -n "$LIMIT" ]]; then
    extra+=(--limit "$LIMIT")
  fi

  log "RUN gen $stem (max_new_tokens=$max_tokens)"
  local model_args=()
  if [[ -n "$MODEL_PATH" ]]; then
    model_args+=(--model_path "$MODEL_PATH")
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VENV" "$BASELINE_SCRIPT" \
    --manifest "$manifest" \
    --mode orig \
    --max_new_tokens "$max_tokens" \
    --out "$out" \
    "${model_args[@]}" \
    "${extra[@]}"
}

run_actionable_judge() {
  local stem="$1"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  if [[ ! -f "$gen" ]]; then
    log "SKIP judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP actionable judge $stem (exists)"
    return 0
  fi
  log "RUN actionable judge $stem"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VENV" "$EVAL/judge/judge_actionable_safety.py" \
    --config "$JUDGE_CFG" \
    --judge_style actionable \
    --generations "$gen" \
    --out "$judged"
}

run_context_judge() {
  local stem="$1"
  local manifest="$2"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  if [[ ! -f "$gen" ]]; then
    log "SKIP context judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP context judge $stem (exists)"
    return 0
  fi
  log "RUN context-aware judge $stem"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VENV" "$EVAL/judge/judge_context_aware.py" \
    --config "$JUDGE_CFG" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$judged"
}

run_over_refusal_judge() {
  local stem="benign_multimodal_n60"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  if [[ ! -f "$gen" ]]; then
    log "SKIP over-refusal judge (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP over-refusal judge (exists)"
    return 0
  fi
  log "RUN over-refusal judge"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VENV" "$EVAL/judge/judge_over_refusal.py" \
    --config "$JUDGE_CFG" \
    --generations "$gen" \
    --out "$judged"
}

run_sciqa_score() {
  local stem="scienceqa_imgval_n200"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/sciqa_score.csv"
  if [[ ! -f "$gen" ]]; then
    log "SKIP sciqa score (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP sciqa score (exists)"
    return 0
  fi
  log "RUN sciqa score (CPU)"
  "$VENV" "$EVAL/score/score_scienceqa.py" \
    --generations "$gen" \
    --manifest "$DATA/manifests/scienceqa_imgval_n200.jsonl" \
    --out "$csv"
}

log "ROOT=$ROOT"
log "VLM=$VLM OUT_DIR=$OUT_DIR GPU=$CUDA_VISIBLE_DEVICES"
log "VENV=$VENV JUDGE_CFG=$JUDGE_CFG"

if [[ "$SKIP_GEN" != "1" ]]; then
  # Cell A/B/C/D + utility manifests
  run_gen "vlsafe_examine_eval"      "$DATA/manifests/vlsafe_examine_eval.jsonl"      256
  run_gen "spa_vl_test_530"          "$DATA/manifests/spa_vl_test_530.jsonl"          192
  run_gen "mmsb_vision_risk_sdtypo"  "$DATA/manifests/mmsb_vision_risk_sdtypo.jsonl"  192
  run_gen "mm_safetybench_300"       "$DATA/manifests/mm_safetybench_300.jsonl"       192
  run_gen "siuo_167"                 "$DATA/manifests/siuo_167.jsonl"                 192
  run_gen "mssbench_unsafe_full"     "$DATA/manifests/mssbench_unsafe_full.jsonl"     192
  run_gen "scienceqa_imgval_n200"    "$DATA/manifests/scienceqa_imgval_n200.jsonl"    192
  run_gen "benign_multimodal_n60"    "$DATA/manifests/benign_multimodal_n60.jsonl"    192
else
  log "SKIP_GEN=1, generation phase skipped"
fi

if [[ "$SKIP_JUDGE" != "1" ]]; then
  run_actionable_judge "vlsafe_examine_eval"
  run_actionable_judge "spa_vl_test_530"
  run_actionable_judge "mmsb_vision_risk_sdtypo"
  run_actionable_judge "mm_safetybench_300"
  run_context_judge "siuo_167" "$DATA/manifests/siuo_167.jsonl"
  run_context_judge "mssbench_unsafe_full" "$DATA/manifests/mssbench_unsafe_full.jsonl"
  run_over_refusal_judge
  run_sciqa_score

  SUMMARY_MD="$OUT_DIR/baseline_summary.md"
  "$VENV" "$EVAL/aggregate/summarize_baseline_metrics.py" \
    --out_dir "$OUT_DIR" \
    --method "baseline" \
    --out_md "$SUMMARY_MD"
  log "Summary written: $SUMMARY_MD"
else
  log "SKIP_JUDGE=1, judging/scoring phase skipped"
fi

log "DONE baseline pipeline for $VLM"
log "Outputs: $OUT_DIR"
