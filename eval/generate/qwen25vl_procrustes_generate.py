#!/usr/bin/env python3
"""Qwen2.5-VL-7B-Instruct generation with Procrustes hooks.

Loads fitted Qwen2.5-VL Procrustes params and applies ProcrustesHookManager on
model.model.layers (28 layers, Qwen2.5 backbone).
"""
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
from procrustes.hooks import ProcrustesHookManager
from procrustes.io import load_params


def load_qwen25vl(model_path: str, dtype=torch.bfloat16):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def get_decoder_layers(model):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def prepare_inputs(processor, query: str, image: Image.Image | None, device):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": query})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": text, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    return processor(**kwargs).to(device)


def generate_one(model, processor, record, mode, max_new_tokens=256) -> str:
    image = None
    if mode == "orig" and record.image_path:
        image = Image.open(record.image_path).convert("RGB")
    inputs = prepare_inputs(processor, record.query, image, model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--mode", default="orig")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_mean", type=float, default=0.0)
    parser.add_argument("--rotation_scale", type=float, default=1.0)
    parser.add_argument("--mean_shift_mode", default="full",
                        choices=["full", "projected_rotation", "refusal_projected"])
    parser.add_argument("--refusal_dir")
    parser.add_argument("--trust_region_rho", type=float, default=None)
    parser.add_argument("--hook_scope", default="prefill_only")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--method_tag", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    model, processor = load_qwen25vl(args.model_path)
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(f"Qwen2.5-VL: {n_layers} decoder layers")

    payload = load_params(args.params)
    params_by_layer = payload["params_by_layer"]
    fitted_layers = sorted(params_by_layer.keys())
    spec = args.layers.strip()
    if spec == "all":
        target_layers = fitted_layers
    elif spec.startswith("last_"):
        target_layers = fitted_layers[-int(spec.split("_", 1)[1]):]
    elif "-" in spec and "," not in spec:
        lo, hi = (int(s) for s in spec.split("-", 1))
        target_layers = [l for l in fitted_layers if lo <= l <= hi]
    else:
        wanted = {int(x) for x in spec.split(",")}
        target_layers = [l for l in fitted_layers if l in wanted]
    target_layers = [l for l in target_layers if l < n_layers]
    print(f"applying Procrustes on {len(target_layers)} layers: {target_layers}")

    method_tag = args.method_tag or "qwen25vl_procrustes_ours"
    refusal_dir_by_layer = {}
    if args.refusal_dir:
        rd_payload = torch.load(args.refusal_dir, map_location="cpu", weights_only=False)
        refusal_dir_by_layer = {int(l): v for l, v in rd_payload["refusal_dir_by_layer"].items()}
        print(f"loaded refusal dir for {len(refusal_dir_by_layer)} layers")
    rows = []
    for record in tqdm(records, desc=method_tag):
        try:
            with ProcrustesHookManager(
                decoder_layers, params_by_layer, target_layers,
                alpha=args.alpha, lambda_mean=args.lambda_mean,
                rotation_scale=args.rotation_scale,
                mean_shift_mode=args.mean_shift_mode,
                refusal_dir_by_layer=refusal_dir_by_layer if refusal_dir_by_layer else None,
                trust_region_rho=args.trust_region_rho,
                hook_scope=args.hook_scope,
            ):
                resp = generate_one(model, processor, record, args.mode, args.max_new_tokens)
        except Exception as e:
            resp = f"<ERROR: {e}>"
        rows.append({
            "id": record.id, "query": record.query, "method": method_tag,
            "input_mode": args.mode, "model_id": args.model_path, "response": resp,
            "alpha": args.alpha, "lambda_mean": args.lambda_mean,
            "rotation_scale": args.rotation_scale,
            "mean_shift_mode": args.mean_shift_mode,
            "layers": args.layers, "hook_scope": args.hook_scope,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
