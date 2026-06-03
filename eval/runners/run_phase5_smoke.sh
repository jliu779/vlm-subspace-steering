#!/usr/bin/env bash
# Phase 5 smoke test: is refp's sciqa destruction a magnitude problem (α too high)
# or a direction problem (wrong r̂)? Try refp with α ∈ {0.1, 0.2} on qwen25vl +
# internvl25 (the two VLMs with worst sciqa collapse under refp at α=0.5:
# qwen25vl 84→18, internvl25 95.5→3.5).
#
# Only sciqa200 + benign60 needed for the magnitude diagnosis. If sciqa recovers
# at α=0.2 or α=0.1, refitting r̂ won't help. If sciqa stays dead, refit may help.
set -u

LANE="${LANE:?LANE=gpu4 or gpu5}"
GPU="${GPU:?GPU index}"

VENV=/home/kedong/repos/VLM-subspace-steering/CMRM/.venv/bin/python
CMRM=/home/kedong/repos/VLM-subspace-steering/CMRM
PR=/home/kedong/repos/VLM-subspace-steering/ProcrustesRotation
OUT_BASE=$PR/outputs/phase5_smoke_2026-05-27

cd $PR

# Lane assignment: gpu4 = qwen25vl × {α=0.1, α=0.2}; gpu5 = internvl25 × {α=0.1, α=0.2}
declare -a configs
if [[ "$LANE" == "gpu4" ]]; then
  configs=("qwen25vl:0.1" "qwen25vl:0.2")
else
  configs=("internvl25:0.1" "internvl25:0.2")
fi

declare -a benches=(
  "sciqa200:$CMRM/data/scienceqa_imgval_n200.jsonl:192"
  "benign60:$CMRM/data/benign_multimodal_n60.jsonl:192"
)

for spec in "${configs[@]}"; do
  vlm="${spec%%:*}"; alpha="${spec##*:}"
  alpha_tag=$(echo $alpha | tr -d '.')

  case "$vlm" in
    qwen25vl)
      gen=scripts/qwen25vl_procrustes_generate.py
      params=$PR/outputs/qwen25vl_procrustes_params_k16.pt
      refdir=$PR/outputs/refusal_dir_qwen25vl.pt
      gen_args=("--manifest")
      ;;
    internvl25)
      gen=scripts/internvl_procrustes_generate.py
      params=$PR/outputs/internvl_procrustes_params_k16.pt
      refdir=$PR/outputs/refusal_dir_internvl.pt
      gen_args=("--manifest")
      ;;
    *)
      echo "unknown vlm $vlm" >&2; exit 2 ;;
  esac

  outdir=$OUT_BASE/${vlm}_a${alpha_tag}_refp
  mkdir -p "$outdir"

  for b in "${benches[@]}"; do
    name="${b%%:*}"; rest="${b#*:}"; manifest="${rest%:*}"; mtoks="${rest##*:}"
    out="$outdir/${name}.jsonl"
    if [[ -f "$out" && -s "$out" ]]; then
      echo "[$(date +%T)] gpu=$GPU SKIP ${vlm}_a${alpha_tag}/${name}"
      continue
    fi
    echo "[$(date +%T)] gpu=$GPU RUN ${vlm}_a${alpha_tag}/${name} (refp α=$alpha)"
    CUDA_VISIBLE_DEVICES=$GPU $VENV "$gen" \
      --manifest "$manifest" --params "$params" \
      --alpha "$alpha" --lambda_mean 1 --mean_shift_mode refusal_projected \
      --refusal_dir "$refdir" \
      --hook_scope prefill_only --max_new_tokens "$mtoks" \
      --out "$out"
    rc=$?
    echo "[$(date +%T)] gpu=$GPU DONE ${vlm}_a${alpha_tag}/${name} rc=$rc"
    if [[ $rc -ne 0 ]]; then
      echo "[$(date +%T)] gpu=$GPU ABORT lane (rc=$rc)" >&2
      exit $rc
    fi
  done
done
echo "[$(date +%T)] gpu=$GPU LANE COMPLETE"
