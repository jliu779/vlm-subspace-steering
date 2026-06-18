"""Phi-4-multimodal-instruct helpers.

Phi-4-multimodal uses Phi-4-Mini-Instruct as the language backbone with vision
and speech adapters. Official inference uses hand-built chat tokens:

    <|user|><|image_1|>{question}<|end|><|assistant|>

See microsoft/Phi-4-multimodal-instruct model card (sample_inference_phi4mm.py).
"""
from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image

_USER = "<|user|>"
_ASSISTANT = "<|assistant|>"
_END = "<|end|>"


def make_blank_pil(size: int = 448, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def get_decoder_layers(model):
    """Phi-4-multimodal CausalLM decoder stack (native or remote-code layout)."""
    inner = model.model
    # Remote modeling_phi4mm may wrap the backbone with PEFT during __init__.
    if hasattr(inner, "get_base_model"):
        inner = inner.get_base_model()
    elif hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model
    return inner.layers


def _from_pretrained_dtype_kwargs(dtype):
    """transformers>=4.56 prefers `dtype`; older builds use `torch_dtype`."""
    return {"dtype": dtype}


def _load_pretrained(model_cls, model_path: str, dtype, device_map: str, attn_implementation: str, **extra):
    kwargs = dict(
        device_map=device_map,
        attn_implementation=attn_implementation,
        **_from_pretrained_dtype_kwargs(dtype),
        **extra,
    )
    try:
        return model_cls.from_pretrained(model_path, **kwargs).eval()
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        return model_cls.from_pretrained(model_path, **kwargs).eval()


def _patch_remote_phi4mm_module() -> bool:
    """PEFT vision LoRA init expects Phi4MMModel.prepare_inputs_for_generation."""
    import sys

    patched = False
    for mod in sys.modules.values():
        if mod is None or not hasattr(mod, "Phi4MMModel"):
            continue
        cls = mod.Phi4MMModel
        if hasattr(cls, "prepare_inputs_for_generation"):
            continue

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}

        cls.prepare_inputs_for_generation = prepare_inputs_for_generation
        patched = True
    return patched


def _preload_and_patch_phi4_remote(model_path: str) -> None:
    """Import cached remote modeling_phi4mm and patch before weight load."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM")
    if class_ref:
        get_class_from_dynamic_module(class_ref, model_path)
    _patch_remote_phi4mm_module()


def load_phi4(
    model_path: str = "/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load Phi-4-multimodal via transformers + AutoProcessor.

    Prefers the in-tree ``Phi4MultimodalForCausalLM`` (no PEFT-in-__init__ issue).
    Falls back to remote ``Phi4MMForCausalLM`` with a PEFT compatibility patch for
    stale cached ``modeling_phi4mm.py`` files missing ``prepare_inputs_for_generation``.

    Defaults to eager attention (no flash-attn requirement). Set
    attn_implementation="flash_attention_2" on Ampere+ if flash-attn is installed.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    try:
        from transformers import Phi4MultimodalForCausalLM

        model = _load_pretrained(
            Phi4MultimodalForCausalLM,
            model_path,
            dtype,
            device_map,
            attn_implementation,
        )
        return model, processor
    except ImportError:
        pass

    _preload_and_patch_phi4_remote(model_path)
    model = _load_pretrained(
        AutoModelForCausalLM,
        model_path,
        dtype,
        device_map,
        attn_implementation,
        trust_remote_code=True,
        _attn_implementation=attn_implementation,
    )
    return model, processor


def build_phi4_prompt(_processor, question: str, has_image: bool) -> str:
    """Return an official-style Phi-4-multimodal prompt string."""
    if has_image:
        body = f"<|image_1|>{question}"
    else:
        body = question
    return f"{_USER}{body}{_END}{_ASSISTANT}"


__all__ = [
    "load_phi4",
    "make_blank_pil",
    "get_decoder_layers",
    "build_phi4_prompt",
]
