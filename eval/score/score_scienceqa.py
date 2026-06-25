#!/usr/bin/env python3
"""Score ScienceQA / MMStar / MME-RW generations: letter match against answer_letter."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Match explicit "answer is/: X" cues (case-insensitive)
_ANSWER_CUE = re.compile(r"answer\s*(?:is|:)?\s*\**\s*\(?([A-Za-z])\b", re.I)
# Match a leading choice like "A." / "A)" / "A " at start of response
_LEADING = re.compile(r"^\s*\(?([A-Za-z])[\).\s]")


def first_letter(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    # 1. Explicit "answer is X" / "answer: X" cue, or leading "X." / "(X)"
    m = _ANSWER_CUE.search(s) or _LEADING.match(s)
    if m:
        return m.group(1).upper()
    # 2. Any isolated UPPERCASE letter only — avoids spurious matches on
    #    English articles ("a") or error-message words ("GiB", "GPU", etc.)
    m = re.search(r"\b([A-Z])\b", s)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, help="qwen3vl scienceqa jsonl")
    ap.add_argument("--manifest", required=True, help="scienceqa manifest with answer_letter")
    ap.add_argument("--out", required=True, help="CSV summary")
    ap.add_argument("--per_sample_out", help="per-sample judgement jsonl")
    args = ap.parse_args()

    gold = {}
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            gold[r["id"]] = r["answer_letter"]

    total = 0
    correct = 0
    parse_err = 0
    per_sample = []
    method = "unknown"
    with open(args.generations) as f:
        for line in f:
            r = json.loads(line)
            method = r.get("method", method)
            rid = r["id"]
            gt = gold.get(rid)
            pred = first_letter(r.get("response", ""))
            total += 1
            if pred is None:
                parse_err += 1
                ok = False
            else:
                ok = (pred == gt)
                if ok: correct += 1
            per_sample.append({"id": rid, "method": method, "gt": gt,
                              "pred": pred, "correct": ok,
                              "response": r.get("response", "")})

    # accuracy denominator = total samples (parse errors count as wrong),
    # consistent with score_mathvista.py
    acc = correct / total if total else 0.0
    summary = {"method": method, "n": total, "correct": correct,
               "parse_error_count": parse_err, "accuracy": acc}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("method,n,correct,parse_error_count,accuracy\n")
        f.write(f"{summary['method']},{summary['n']},{summary['correct']},{summary['parse_error_count']},{summary['accuracy']:.4f}\n")
    if args.per_sample_out:
        Path(args.per_sample_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample_out, "w") as f:
            for r in per_sample:
                f.write(json.dumps(r) + "\n")
    print(f"  {method}: acc={acc:.3f} ({correct}/{total}), parse_err={parse_err}")


if __name__ == "__main__":
    main()
