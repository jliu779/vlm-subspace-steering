#!/usr/bin/env bash
# Phase 5 full-extend: refp α=0.2 on the 4 missing harm cells (B, C/mmsb, D/siuo,
# D/mssb) for one VLM, on one GPU. Confirms refp α=0.2 generalizes beyond Cell A
# + Cell C/sdtypo where it already strictly dominates λ=0.
#
# Usage:  VLM=qwen25vl GPU=4 bash scripts/run_phase5_full_extend.sh
set -u

VLM="${VLM:?VLM=qwen25vl|internvl25|qwen3vl|internvl3}"
GPU="${GPU:?GPU index}"

VENV=/home/kedong/repos/VLM-subspace-steering/CMRM/.venv/bin/python
CMRM=/home/kedong/repos/VLM-subspace-steering/CMRM
PR=/home/kedong/repos/VLM-subspace-steering/ProcrustesRotation
OUT_BASE=$PR/outputs/phase5_full_extend_2026-05-28
MNT=192

cd $PR

case "$VLM" in
  qwen25vl)
    GEN=scripts/qwen25vl_procrustes_generate.py
    PARAMS=$PR/outputs/qwen25vl_procrustes_params_k16.pt
    REFDIR=$PR/outputs/refusal_dir_qwen25vl.pt
    ;;
  internvl25)
    GEN=scripts/internvl_procrustes_generate.py
    PARAMS=$PR/outputs/internvl_procrustes_params_k16.pt
    REFDIR=$PR/outputs/refusal_dir_internvl.pt
    ;;
  qwen3vl)
    GEN=scripts/qwen3vl_procrustes_generate.py
    PARAMS=$PR/outputs/qwen3vl_procrustes_params_k16.pt
    REFDIR=$PR/outputs/refusal_dir_qwen3vl.pt
    ;;
  internvl3)
    GEN=scripts/internvl3_procrustes_generate.py
    PARAMS=$PR/outputs/internvl3_procrustes_params_k16.pt
    REFDIR=$PR/outputs/refusal_dir_internvl3.pt
    ;;
  *) echo "unknown VLM: $VLM" >&2; exit 1;;
esac

declare -a benches=(
  "spa_vl_test_530:$CMRM/data/spa_vl_test_530.jsonl"
  "mm_safetybench_300:$CMRM/data/mm_safetybench_300.jsonl"
  "siuo_167:$CMRM/data/siuo_167.jsonl"
  "mssbench_unsafe_full:$CMRM/data/mssbench_unsafe_full.jsonl"
)

outdir=$OUT_BASE/${VLM}_a02_refp
mkdir -p "$outdir"

for b in "${benches[@]}"; do
  name="${b%%:*}"; manifest="${b##*:}"
  out="$outdir/${name}.jsonl"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "[$(date +%T)] gpu=$GPU SKIP ${VLM}_a02/${name}"
    continue
  fi
  echo "[$(date +%T)] gpu=$GPU RUN ${VLM}_a02/${name}"
  CUDA_VISIBLE_DEVICES=$GPU $VENV "$GEN" \
    --manifest "$manifest" --params "$PARAMS" \
    --alpha 0.2 --lambda_mean 1 --mean_shift_mode refusal_projected \
    --refusal_dir "$REFDIR" \
    --hook_scope prefill_only --max_new_tokens "$MNT" \
    --out "$out"
  rc=$?
  echo "[$(date +%T)] gpu=$GPU DONE ${VLM}_a02/${name} rc=$rc"
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%T)] gpu=$GPU ABORT lane (rc=$rc)" >&2
    exit $rc
  fi
done
echo "[$(date +%T)] gpu=$GPU ${VLM} FULL_EXTEND COMPLETE"
