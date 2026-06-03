from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def parse_dtype(dtype: str) -> str:
    allowed = {"float16", "fp16", "bfloat16", "bf16", "float32", "fp32", "auto"}
    if dtype not in allowed:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {sorted(allowed)}")
    return {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(dtype, dtype)


def torch_dtype(dtype: str):
    dtype = parse_dtype(dtype)
    if dtype == "auto":
        return "auto"
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def parse_layers(spec: str | int | list[int] | None, n_layers: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(n_layers))
    if isinstance(spec, int):
        spec = str(spec)
    if isinstance(spec, list):
        layers = spec
    else:
        layers = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = [int(x) for x in part.split("-", 1)]
                layers.extend(range(start, end + 1))
            else:
                layers.append(int(part))
    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"layer index out of range for {n_layers} layers: {bad}")
    return sorted(dict.fromkeys(layers))
