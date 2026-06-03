#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metrics(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return "", ""
    return rows[0].get("unsafe_count", ""), rows[0].get("unsafe_rate", "")


def dominates(a: dict, b: dict) -> bool:
    # Minimize unsafe_rate and short_le3_ratio.
    ua = float(a["unsafe_rate"]) if a["unsafe_rate"] else 1e9
    ub = float(b["unsafe_rate"]) if b["unsafe_rate"] else 1e9
    sa = float(a["short_le3_ratio"])
    sb = float(b["short_le3_ratio"])
    return (ua <= ub and sa <= sb) and (ua < ub or sa < sb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--glob", default="procrustes_*.jsonl")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_pareto_csv", required=True)
    args = parser.parse_args()

    root = Path(args.dir)
    rows = []
    for gen_path in sorted(root.glob(args.glob)):
        name = gen_path.name
        if ".judged." in name:
            continue
        tag = gen_path.stem
        data = list(read_jsonl(gen_path))
        if not data:
            continue
        toks = [len(str(r.get("response", "")).split()) for r in data]
        short = sum(1 for t in toks if t <= 3)
        unsafe_count, unsafe_rate = load_metrics(gen_path.with_suffix(".metrics.csv"))
        rows.append(
            {
                "tag": tag,
                "n": len(data),
                "unsafe_count": unsafe_count,
                "unsafe_rate": unsafe_rate,
                "short_le3_count": short,
                "short_le3_ratio": f"{short / len(data):.6f}",
                "mean_tokens": f"{mean(toks):.4f}",
            }
        )

    rows.sort(key=lambda x: x["tag"])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pareto = []
    for r in rows:
        if any(dominates(o, r) for o in rows if o is not r):
            continue
        pareto.append(r)
    pareto.sort(
        key=lambda x: (
            float(x["unsafe_rate"]) if x["unsafe_rate"] else 1e9,
            float(x["short_le3_ratio"]),
        )
    )
    out_pareto = Path(args.out_pareto_csv)
    with out_pareto.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(pareto)

    print(f"wrote summary: {out_csv}")
    print(f"wrote pareto : {out_pareto}")


if __name__ == "__main__":
    main()

