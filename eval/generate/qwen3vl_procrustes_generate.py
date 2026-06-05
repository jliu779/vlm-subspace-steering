#!/usr/bin/env python3
"""Qwen3-VL-8B-Instruct generation with Procrustes hooks (Ours method).

Loads fitted Qwen Procrustes params and applies ProcrustesHookManager on
model.model.language_model.layers (36 layers).
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


def load_qwen3vl(model_path: str, dtype=torch.bfloat16):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def get_decoder_layers(model):
    return model.model.language_model.layers


def prepare_inputs(processor, query: str, image: Image.Image | None, device):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": query})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
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
    parser.add_argument("--model_path", default="/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--params", required=True, help="fitted Qwen Procrustes params .pt")
    parser.add_argument("--mode", default="orig")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_mean", type=float, default=1.0)
    parser.add_argument("--rotation_scale", type=float, default=1.0,
                        help="Multiplier on the rotation (-p_c + p_t) term. "
                             "Set to 0 for mean-only ablation.")
    parser.add_argument("--mean_shift_mode", default="full",
                        choices=["full", "projected_rotation", "refusal_projected"],
                        help="'full': lambda_mean * (mu_t - mu_c). "
                             "'projected_rotation': project mean shift onto "
                             "per-sample rotation correction direction. "
                             "'refusal_projected': project mean shift onto "
                             "per-layer refusal direction r̂; requires --refusal_dir.")
    parser.add_argument("--refusal_dir",
                        help="Optional refusal-direction .pt with key 'refusal_dir_by_layer'. "
                             "Used when --mean_shift_mode=refusal_projected.")
    parser.add_argument("--trust_region_rho", type=float, default=None,
                        help="If set, clip ||delta|| <= rho * ||h||. "
                             "Recommended 0.03 for Qwen3-VL.")
    parser.add_argument("--hook_scope", default="prefill_only")
    parser.add_argument("--mean_shift_gate_threshold", type=float, default=None)
    parser.add_argument("--mean_shift_gate_temp", type=float, default=0.05)
    parser.add_argument("--layers", default="all",
                        help="all | last_N (e.g. last_8) | A-B (e.g. 28-35) | "
                             "comma list (e.g. 0,5,10)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--method_tag", default=None,
                        help="Override method tag in output. Default 'qwen3vl_procrustes_ours'.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    model, processor = load_qwen3vl(args.model_path)
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(f"Qwen3-VL: {n_layers} decoder layers")

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

    method_tag = args.method_tag or "qwen3vl_procrustes_ours"
    refusal_dir_by_layer = {}
    if args.refusal_dir:
        import torch as _torch
        rd_payload = _torch.load(args.refusal_dir, map_location="cpu", weights_only=False)
        refusal_dir_by_layer = {int(l): v for l, v in rd_payload["refusal_dir_by_layer"].items()}
        print(f"loaded refusal dir for {len(refusal_dir_by_layer)} layers from {args.refusal_dir}")
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
                mean_shift_gate_threshold=args.mean_shift_gate_threshold,
                mean_shift_gate_temp=args.mean_shift_gate_temp,
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
            "mean_shift_gate_threshold": args.mean_shift_gate_threshold,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
