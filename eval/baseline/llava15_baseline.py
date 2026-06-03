#!/usr/bin/env python3
"""Generate baseline (no intervention) responses for the eval manifest, in any
of the standard CMRM input modes. Mirrors CMRM/scripts/03_generate_baseline.py
exactly so we can reuse CMRM's existing baseline generations if available."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from procrustes.cmrm_compat import (
    first_model_device,
    load_config,
    load_vlm,
    prepare_inputs,
    read_manifest,
    write_jsonl,
)


def generate_one(model, processor, record, mode: str, cfg, image_root=None, generation_overrides=None) -> str:
    import torch

    device = first_model_device(model)
    inputs = prepare_inputs(processor, record, mode, image_root=image_root, device=device)
    gen_cfg = dict(cfg.get("generation", {}))
    if generation_overrides:
        gen_cfg.update(generation_overrides)
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_cfg)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--modes", nargs="+", default=["orig"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_root = cfg.get("paths", {}).get("image_root")
    records = read_manifest(args.manifest, limit=args.limit)
    model, processor = load_vlm(cfg)

    rows = []
    generation_overrides = {}
    if args.max_new_tokens is not None:
        generation_overrides["max_new_tokens"] = args.max_new_tokens
    for record in tqdm(records, desc="baseline"):
        for mode in args.modes:
            rows.append({
                "id": record.id,
                "query": record.query,
                "method": "baseline",
                "input_mode": mode,
                "model_id": cfg["model"]["target_model_id"],
                "response": generate_one(
                    model,
                    processor,
                    record,
                    mode,
                    cfg,
                    image_root=image_root,
                    generation_overrides=generation_overrides,
                ),
            })
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} baseline rows to {args.out}")


if __name__ == "__main__":
    main()
