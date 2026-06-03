"""Persistence for fitted Procrustes parameters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_params(params_by_layer: dict[int, dict], meta: dict[str, Any], out: str | Path) -> None:
    import torch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params_by_layer": {int(k): v for k, v in params_by_layer.items()}, "meta": meta}
    torch.save(payload, out)


def load_params(path: str | Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["params_by_layer"] = {int(k): v for k, v in payload["params_by_layer"].items()}
    return payload


def save_meta_json(meta: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
