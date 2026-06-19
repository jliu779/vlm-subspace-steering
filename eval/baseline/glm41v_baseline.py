#!/usr/bin/env python3
"""GLM-4.1V-9B-Thinking baseline generation, no intervention."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from procrustes.cmrm_compat import read_manifest
from procrustes.glm41v_utils import build_glm41v_messages, load_glm41v


def _looks_like_cache_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "cache",
            "past_key",
            "past key",
            "past_key_values",
            "get_usable_length",
            "get_max_length",
        )
    )


def generate_one(
    model,
    processor,
    record,
    mode: str,
    max_new_tokens: int = 256,
    use_cache: bool = True,
) -> str:
    image_path = record.image_path if mode == "orig" and record.image_path else None
    messages = build_glm41v_messages(record.query, image_path)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            repetition_penalty=1.1,
        )
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[0, prompt_len:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", default="orig", choices=["orig", "query_only"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="GLM-4.1V attention backend; flash_attention_2 requires flash-attn.",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Compatibility fallback: disables KV cache, but generation is much slower.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = read_manifest(args.manifest, limit=args.limit)
    use_cache = not args.disable_cache
    model, processor = load_glm41v(
        args.model_path,
        attn_implementation=args.attn_implementation,
        use_cache=use_cache,
    )
    method_tag = "glm41v_baseline"

    rows = []
    for record in tqdm(records, desc=method_tag):
        try:
            resp = generate_one(
                model,
                processor,
                record,
                args.mode,
                args.max_new_tokens,
                use_cache=use_cache,
            )
        except Exception as e:
            if use_cache and _looks_like_cache_error(e):
                print(
                    f"WARNING: GLM-4.1V KV cache failed ({e}); retrying with cache disabled.",
                    file=sys.stderr,
                )
                use_cache = False
                model.config.use_cache = False
                try:
                    resp = generate_one(
                        model,
                        processor,
                        record,
                        args.mode,
                        args.max_new_tokens,
                        use_cache=use_cache,
                    )
                except Exception as retry_e:
                    resp = f"<ERROR: {retry_e}>"
            else:
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
