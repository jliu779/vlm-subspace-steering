#!/usr/bin/env bash
# Phase 5 extension: test α=0.2 refp on qwen3vl + internvl3 (the other 2 VLMs
# whose §5 Cell A pick is currently λ=0). If refp α=0.2 also Pareto-dominates
# λ=0 here, the magnitude finding generalizes across 4 VLMs.
#
# Lane: gpu4 = qwen3vl, gpu5 = internvl3. Each lane runs sciqa200, benign60,
# vlsafe1110, mmsb_sdtypo with refp α=0.2.
set -u

LANE="${LANE:?LANE=gpu4 or gpu5}"
GPU="${GPU:?GPU index}"

VENV=/home/kedong/repos/VLM-subspace-steering/CMRM/.venv/bin/python
CMRM=/home/kedong/repos/VLM-subspace-steering/CMRM
PR=/home/kedong/repos/VLM-subspace-steering/ProcrustesRotation
OUT_BASE=$PR/outputs/phase5_extend_2026-05-28

cd $PR

if [[ "$LANE" == "gpu4" ]]; then
  vlm=qwen3vl
  gen=scripts/qwen3vl_procrustes_generate.py
  params=$PR/outputs/qwen3vl_procrustes_params_k16.pt
  refdir=$PR/outputs/refusal_dir_qwen3vl.pt
else
  vlm=internvl3
  gen=scripts/internvl3_procrustes_generate.py
  params=$PR/outputs/internvl3_procrustes_params_k16.pt
  refdir=$PR/outputs/refusal_dir_internvl3.pt
fi

declare -a benches=(
  "sciqa200:$CMRM/data/scienceqa_imgval_n200.jsonl:192"
  "benign60:$CMRM/data/benign_multimodal_n60.jsonl:192"
  "vlsafe1110:$CMRM/data/vlsafe_examine_eval.jsonl:256"
  "mmsb_sdtypo:$CMRM/data/mmsb_vision_risk_sdtypo.jsonl:192"
)

outdir=$OUT_BASE/${vlm}_a02_refp
mkdir -p "$outdir"

for b in "${benches[@]}"; do
  name="${b%%:*}"; rest="${b#*:}"; manifest="${rest%:*}"; mtoks="${rest##*:}"
  out="$outdir/${name}.jsonl"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "[$(date +%T)] gpu=$GPU SKIP ${vlm}_a02/${name}"
    continue
  fi
  echo "[$(date +%T)] gpu=$GPU RUN ${vlm}_a02/${name} (refp α=0.2)"
  CUDA_VISIBLE_DEVICES=$GPU $VENV "$gen" \
    --manifest "$manifest" --params "$params" \
    --alpha 0.2 --lambda_mean 1 --mean_shift_mode refusal_projected \
    --refusal_dir "$refdir" \
    --hook_scope prefill_only --max_new_tokens "$mtoks" \
    --out "$out"
  rc=$?
  echo "[$(date +%T)] gpu=$GPU DONE ${vlm}_a02/${name} rc=$rc"
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%T)] gpu=$GPU ABORT lane (rc=$rc)" >&2
    exit $rc
  fi
done
echo "[$(date +%T)] gpu=$GPU LANE COMPLETE"
