#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from cmrm.config import load_config
from cmrm.generation import read_jsonl, write_jsonl
from cmrm.judging import (
    JUDGE_SYSTEM_PROMPT_ACTIONABLE,
    JUDGE_SYSTEM_PROMPT_REFUSAL,
    judge_one,
)
from cmrm.models import load_judge


_JUDGE_STYLES = {
    "actionable": JUDGE_SYSTEM_PROMPT_ACTIONABLE,
    "refusal": JUDGE_SYSTEM_PROMPT_REFUSAL,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--judge_style",
        choices=list(_JUDGE_STYLES),
        default="actionable",
        help=(
            "Which judge prompt to use. 'actionable' (default) marks UNSAFE only "
            "when the response actually enables harm; matches the looser "
            "interpretation that produces paper-style numbers. 'refusal' marks "
            "UNSAFE whenever the assistant fails to clearly refuse; reproduces "
            "the strict-refusal numbers from earlier rounds."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    system_prompt = _JUDGE_STYLES[args.judge_style]
    model, tokenizer = load_judge(cfg)
    rows = []
    for row in tqdm(read_jsonl(args.generations), desc="judge"):
        label, raw = judge_one(
            model,
            tokenizer,
            row.get("query", ""),
            row.get("response", ""),
            system_prompt=system_prompt,
        )
        rows.append(
            {
                "id": row.get("id"),
                "method": row.get("method"),
                "input_mode": row.get("input_mode"),
                "label": label,
                "judge_raw": raw,
                "judge_style": args.judge_style,
                "judge_model_id": cfg["model"]["judge_model_id"],
            }
        )
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} labels to {args.out}")


if __name__ == "__main__":
    main()
