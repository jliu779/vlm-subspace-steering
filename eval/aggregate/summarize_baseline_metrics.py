#!/usr/bin/env python3
"""Summarize baseline (or any method) judged outputs into a markdown table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unsafe_rate(rows: list[dict]) -> tuple[int, int, float]:
    valid = [r for r in rows if r.get("label") in {"UNSAFE", "SAFE"}]
    if not valid:
        return 0, 0, 0.0
    unsafe = sum(1 for r in valid if r["label"] == "UNSAFE")
    return unsafe, len(valid), unsafe / len(valid)


def over_refusal_rate(rows: list[dict]) -> tuple[int, int, float]:
    # In over-refusal judge: UNSAFE means over-refused.
    return unsafe_rate(rows)


def read_sciqa_csv(path: Path) -> str:
    if not path.exists():
        return "N/A"
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "N/A"
    acc = rows[0].get("overall_accuracy") or rows[0].get("accuracy")
    return f"{float(acc) * 100:.2f}%" if acc is not None else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="e.g. outputs/qwen25vl_baseline")
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    benches = [
        ("Cell A", "vlsafe_examine_eval", "actionable"),
        ("Cell B", "spa_vl_test_530", "actionable"),
        ("Cell C/sdtypo", "mmsb_vision_risk_sdtypo", "actionable"),
        ("Cell C/MMSB", "mm_safetybench_300", "actionable"),
        ("Cell D/SIUO", "siuo_167", "context"),
        ("Cell D/MSSB", "mssbench_unsafe_full", "context"),
        ("Utility sciqa", "scienceqa_imgval_n200", "sciqa"),
        ("Utility OR", "benign_multimodal_n60", "over_refusal"),
    ]

    lines = [
        f"# {args.method} summary — `{out_dir.name}`",
        "",
        "| Benchmark | Metric | Value |",
        "|---|---|---|",
    ]

    for label, stem, kind in benches:
        if kind == "sciqa":
            val = read_sciqa_csv(out_dir / "sciqa_score.csv")
            lines.append(f"| {label} | sciqa% | {val} |")
            continue

        judged = out_dir / f"{stem}.judged.jsonl"
        if not judged.exists():
            lines.append(f"| {label} | — | MISSING |")
            continue

        rows = read_jsonl(judged)
        if kind == "over_refusal":
            n_bad, n, rate = over_refusal_rate(rows)
            metric = "OR%"
            val = f"{rate * 100:.2f}% ({n_bad}/{n})"
        else:
            n_unsafe, n, rate = unsafe_rate(rows)
            metric = "unsafe%"
            val = f"{rate * 100:.2f}% ({n_unsafe}/{n})"
        lines.append(f"| {label} | {metric} | {val} |")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
