from __future__ import annotations

from typing import Any

from .assets import resolve_model_id
from .config import torch_dtype


def load_vlm(cfg: dict[str, Any]):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = resolve_model_id(
        cfg["model"]["target_model_id"],
        cfg.get("paths", {}).get("model_root", "/hub/huggingface/models"),
    )
    trust_remote = cfg["model"].get("trust_remote_code", False)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote)
    dtype = torch_dtype(cfg["model"].get("dtype", "float16"))
    device = cfg["model"].get("device_map", "auto")

    # Try standard LlavaForConditionalGeneration first; fall back to AutoModel
    # for custom architectures (e.g. ShareGPT4V's share4v type).
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote,
        )
    except (ValueError, OSError):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote,
        )
    model.eval()
    return model, processor


def load_judge(cfg: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve_model_id(
        cfg["model"]["judge_model_id"],
        cfg.get("paths", {}).get("model_root", "/hub/huggingface/models"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(cfg["model"].get("dtype", "float16")),
        device_map=cfg["model"].get("device_map", "auto"),
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def get_decoder_layers(model) -> list[Any]:
    paths = [
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "layers"),
    ]
    for path in paths:
        node = model
        ok = True
        for part in path:
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok:
            return list(node)
    children = [name for name, _ in getattr(model, "named_children", lambda: [])()]
    raise ValueError(f"cannot find decoder layers; top-level children: {children}")


def first_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"
