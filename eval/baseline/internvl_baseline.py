#!/usr/bin/env python3
"""InternVL3.5-8B baseline generation on a multimodal manifest, no intervention."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from tqdm import tqdm

from procrustes.cmrm_compat import read_manifest
from procrustes.internvl_utils import load_internvl, load_image_pixel_values


def generate_one(model, tokenizer, record, mode: str, max_new_tokens: int = 256) -> str:
    pixel_values = None
    question = record.query
    if mode == "orig" and record.image_path:
        pixel_values = load_image_pixel_values(record.image_path, max_num=12).to(model.device)
    gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if pixel_values is not None:
        response = model.chat(tokenizer, pixel_values, question, gen_config)
    else:
        response = model.chat(tokenizer, None, question, gen_config)
    return response.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/hub/huggingface/models/OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", default="orig", choices=["orig", "query_only"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    model, tokenizer = load_internvl(args.model_path)
    method_tag = "internvl_baseline"

    rows = []
    for record in tqdm(records, desc=method_tag):
        try:
            resp = generate_one(model, tokenizer, record, args.mode, args.max_new_tokens)
        except Exception as e:
            resp = f"<ERROR: {e}>"
        rows.append({
            "id": record.id,
            "query": record.query,
            "method": method_tag,
            "input_mode": args.mode,
            "model_id": args.model_path,
            "response": resp,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} {method_tag} rows to {args.out}")


if __name__ == "__main__":
    main()
