from __future__ import annotations

from pathlib import Path
from typing import Any

from .images import load_image, make_blank_image_like, make_noise_image_like
from .manifest import ManifestRecord, resolve_image_path
from .models import first_model_device, get_decoder_layers
from .prompts import build_prompt


def _to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def build_image_for_mode(record: ManifestRecord, mode: str, image_root: str | Path | None, seed: int = 0):
    if mode in {"caption", "query_only"}:
        return None
    path = resolve_image_path(record, image_root)
    if path is None:
        raise ValueError(f"record {record.id} has no image_path for mode {mode}")
    image = load_image(path)
    if mode == "blank":
        return make_blank_image_like(image)
    if mode == "noise":
        return make_noise_image_like(image, seed=seed)
    return image


def prepare_inputs(processor, record: ManifestRecord, mode: str, image_root=None, device=None) -> dict[str, Any]:
    prompt = build_prompt(record.query, mode, record.caption)
    image = build_image_for_mode(record, mode, image_root, seed=abs(hash(record.id)) % (2**31))
    kwargs = {"text": prompt, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    batch = processor(**kwargs)
    return _to_device(batch, device) if device is not None else batch


def decoder_hidden_states(outputs, n_layers: int):
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("model did not return hidden_states")
    if len(hidden_states) < n_layers:
        raise ValueError(f"expected at least {n_layers} hidden states, got {len(hidden_states)}")
    return hidden_states[-n_layers:]


def extract_last_token_hidden(model, processor, record: ManifestRecord, mode: str, image_root=None) -> dict[int, Any]:
    import torch

    device = first_model_device(model)
    inputs = prepare_inputs(processor, record, mode, image_root=image_root, device=device)
    n_layers = len(get_decoder_layers(model))
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    states = decoder_hidden_states(outputs, n_layers)
    return {i: state[:, -1, :].detach().float().cpu().squeeze(0) for i, state in enumerate(states)}


def extract_delta(
    model,
    processor,
    record: ManifestRecord,
    image_root=None,
    corrupted_mode: str = "blank",
    max_norm: float = 0.0,
    per_layer_max_norm: dict | None = None,
    unit_normalize: bool = False,
) -> dict[int, Any]:
    """Compute (h_t - h_c) for each decoder layer.

    Args:
      max_norm: scalar L2 cap applied to every layer (legacy). 0 disables.
      per_layer_max_norm: dict {layer_id: max_norm_for_this_layer}, takes
        priority over `max_norm`. Use this when `max_norm` derived from a
        single global value over-clips deep layers (where natural h_t - h_c
        norm is much larger) and under-clips shallow layers.
      unit_normalize: if True, normalize each layer's delta to unit L2 norm.
        This makes sample-level v geometrically equivalent to dataset-level
        unit_pc (a unit direction times alpha) and is our hypothesis for why
        the CMRM paper reports sample-level < dataset-level. Mutually
        exclusive with the cap arguments — when set, max_norm and
        per_layer_max_norm are ignored.
    """
    text = extract_last_token_hidden(model, processor, record, "query_only", image_root=image_root)
    corrupted = extract_last_token_hidden(model, processor, record, corrupted_mode, image_root=image_root)
    deltas = {layer: text[layer] - corrupted[layer] for layer in text}

    if unit_normalize:
        for layer, vec in deltas.items():
            n = float(vec.norm().item())
            if n > 0:
                deltas[layer] = vec / n
        return deltas

    if per_layer_max_norm:
        for layer, vec in deltas.items():
            cap = per_layer_max_norm.get(layer)
            if cap is None or cap <= 0:
                continue
            n = float(vec.norm().item())
            if n > cap:
                deltas[layer] = vec * (cap / n)
    elif max_norm and max_norm > 0:
        for layer, vec in deltas.items():
            n = float(vec.norm().item())
            if n > max_norm:
                deltas[layer] = vec * (max_norm / n)
    return deltas
