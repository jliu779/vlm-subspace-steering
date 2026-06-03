#!/usr/bin/env python3
"""Gemma-3-4B-IT generation with Procrustes-MPC hooks.

Loads fitted Gemma-3 Procrustes params and applies ProcrustesHookManager on
model.model.language_model.layers (34 layers, hidden 2560).
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
from procrustes.gemma3_utils import load_gemma3, get_decoder_layers, build_gemma3_prompt


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
    return processor.tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/hub/huggingface/models/google/gemma-3-4b-it")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--params", required=True, help="fitted Gemma-3 Procrustes params .pt")
    parser.add_argument("--mode", default="orig")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_mean", type=float, default=1.0)
    parser.add_argument("--rotation_scale", type=float, default=1.0)
    parser.add_argument("--mean_shift_mode", default="full",
                        choices=["full", "projected_rotation", "refusal_projected"])
    parser.add_argument("--refusal_dir")
    parser.add_argument("--trust_region_rho", type=float, default=None)
    parser.add_argument("--hook_scope", default="prefill_only")
    parser.add_argument("--layers", default="all",
                        help="all | last_N | A-B | comma list")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--method_tag", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    model, processor = load_gemma3(args.model_path)
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(f"Gemma-3-4B: {n_layers} decoder layers")

    payload = load_params(args.params)
    params_by_layer = payload["params_by_layer"]
    fitted_layers = sorted(params_by_layer.keys())
    spec = args.layers.strip()
    if spec == "all":
        target_layers = fitted_layers
    elif spec.startswith("last_"):
        n_last = int(spec.split("_", 1)[1])
        target_layers = fitted_layers[-n_last:]
    elif "-" in spec and "," not in spec:
        lo, hi = (int(s) for s in spec.split("-", 1))
        target_layers = [l for l in fitted_layers if lo <= l <= hi]
    else:
        wanted = {int(x) for x in spec.split(",")}
        target_layers = [l for l in fitted_layers if l in wanted]
    target_layers = [l for l in target_layers if l < n_layers]
    print(f"applying Procrustes on {len(target_layers)} layers: {target_layers}")

    method_tag = args.method_tag or "gemma3_procrustes_ours"
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
            "id": record.id,
            "query": record.query,
            "method": method_tag,
            "input_mode": args.mode,
            "model_id": args.model_path,
            "response": resp,
            "alpha": args.alpha,
            "lambda_mean": args.lambda_mean,
            "rotation_scale": args.rotation_scale,
            "mean_shift_mode": args.mean_shift_mode,
            "trust_region_rho": args.trust_region_rho,
            "layers": args.layers,
            "hook_scope": args.hook_scope,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
