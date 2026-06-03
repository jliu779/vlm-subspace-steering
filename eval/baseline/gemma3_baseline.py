#!/usr/bin/env python3
"""Gemma-3-4B-IT baseline generation, no intervention."""
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
from procrustes.gemma3_utils import load_gemma3, build_gemma3_prompt


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_one(model, processor, record, mode: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    has_image = (mode == "orig" and bool(record.image_path))
    prompt = build_gemma3_prompt(processor, record.query, has_image=has_image)

    if has_image:
        img = _load_image(record.image_path)
        inputs = processor(text=prompt, images=[img], return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[0, prompt_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/hub/huggingface/models/google/gemma-3-4b-it")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", default="orig", choices=["orig", "query_only"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    model, processor = load_gemma3(args.model_path)
    method_tag = "gemma3_baseline"

    rows = []
    for record in tqdm(records, desc=method_tag):
        try:
            resp = generate_one(model, processor, record, args.mode, args.max_new_tokens)
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
