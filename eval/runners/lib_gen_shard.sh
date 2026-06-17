# Shared dual-GPU generation sharding (sourced by run_*_full.sh).
# Requires: VENV, log(), OUT_DIR, CUDA_VISIBLE_DEVICES, JUDGE_GPU, DUAL_GPU, LIMIT, GEN_SHARD.

shard_count_lines() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo 0
    return 0
  fi
  wc -l < "$f" | tr -d ' '
}

should_gen_shard() {
  local manifest="$1"
  case "${GEN_SHARD:-auto}" in
    0|off|false|no) return 1 ;;
    2|auto|on|true|yes) ;;
    *) return 1 ;;
  esac
  [[ "${DUAL_GPU:-0}" == "1" ]] || return 1
  [[ -z "${LIMIT:-}" ]] || return 1
  if [[ "${GEN_SHARD:-auto}" == "auto" ]]; then
    local n
    n="$(shard_count_lines "$manifest")"
    [[ "$n" -ge "${SHARD_MIN_LINES:-500}" ]] || return 1
  fi
  return 0
}

shard_split_manifest_halves() {
  local manifest="$1"
  local out0="$2"
  local out1="$3"
  "$VENV" - <<'PY' "$manifest" "$out0" "$out1"
import json
import sys
from pathlib import Path

manifest, out0, out1 = sys.argv[1:4]
rows = [
    json.loads(line)
    for line in Path(manifest).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
mid = len(rows) // 2
if mid == 0 and len(rows) > 0:
    mid = 1
Path(out0).parent.mkdir(parents=True, exist_ok=True)
Path(out0).write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:mid]),
    encoding="utf-8",
)
Path(out1).write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[mid:]),
    encoding="utf-8",
)
print(f"split {len(rows)} -> {len(rows[:mid])} + {len(rows[mid:])}")
PY
}

shard_merge_jsonl() {
  local out="$1"
  local part0="$2"
  local part1="$3"
  cat "$part0" "$part1" > "$out"
}
