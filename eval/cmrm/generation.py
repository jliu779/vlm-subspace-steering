from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import parse_layers
from .hidden import prepare_inputs
from .hooks import CMRMHookManager
from .models import first_model_device, get_decoder_layers


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def decode_new_tokens(processor, generated, inputs) -> str:
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, prompt_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def generate_one(
    model,
    processor,
    record,
    mode: str,
    cfg: dict[str, Any],
    image_root=None,
    generation_overrides: dict[str, Any] | None = None,
) -> str:
    import torch

    device = first_model_device(model)
    inputs = prepare_inputs(processor, record, mode, image_root=image_root, device=device)
    gen_cfg = dict(cfg.get("generation", {}))
    if generation_overrides:
        gen_cfg.update(generation_overrides)
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_cfg)
    return decode_new_tokens(processor, generated, inputs)


def generate_one_cmrm(
    model,
    processor,
    record,
    mode: str,
    cfg: dict[str, Any],
    vectors: dict[int, Any],
    alpha: float,
    sign: int,
    layers_spec="all",
    hook_scope: str = "all_steps",
    image_root=None,
    generation_overrides: dict[str, Any] | None = None,
) -> str:
    layers = get_decoder_layers(model)
    target_layers = parse_layers(layers_spec, len(layers))
    with CMRMHookManager(layers, vectors, target_layers, alpha=alpha, sign=sign, hook_scope=hook_scope):
        return generate_one(
            model,
            processor,
            record,
            mode,
            cfg,
            image_root=image_root,
            generation_overrides=generation_overrides,
        )
