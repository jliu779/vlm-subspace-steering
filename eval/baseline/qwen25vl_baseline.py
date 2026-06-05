#!/usr/bin/env python3
"""Qwen2.5-VL-7B-Instruct baseline generation (no intervention)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from PIL import Image

from procrustes.cmrm_compat import read_manifest


def load_qwen25vl(model_path: str, dtype=torch.bfloat16):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_messages(query: str, image_path: str | None) -> list[dict]:
    content = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": query})
    return [{"role": "user", "content": content}]


def generate_one(model, processor, record, mode: str, max_new_tokens: int = 256) -> str:
    image = None
    if mode == "orig" and record.image_path:
        image = Image.open(record.image_path).convert("RGB")
    messages = build_messages(record.query, record.image_path if image is not None else None)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": text, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    inputs = processor(**kwargs).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", default="orig", choices=["orig", "query_only"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    model, processor = load_qwen25vl(args.model_path)
    method_tag = "qwen25vl_baseline"

    rows = []
    for record in tqdm(records, desc=method_tag):
        try:
            resp = generate_one(model, processor, record, args.mode, args.max_new_tokens)
        except Exception as e:
            resp = f"<ERROR: {e}>"
        rows.append({
            "id": record.id, "query": record.query, "method": method_tag,
            "input_mode": args.mode, "model_id": args.model_path, "response": resp,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
