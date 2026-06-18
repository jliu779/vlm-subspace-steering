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
    """Phi-4-multimodal CausalLM: model.model.layers."""
    return model.model.layers


def load_phi4(
    model_path: str = "/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load Phi-4-multimodal via AutoModelForCausalLM + AutoProcessor.

    Defaults to eager attention (no flash-attn requirement). Set
    attn_implementation="flash_attention_2" on Ampere+ if flash-attn is installed.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_implementation,
        _attn_implementation=attn_implementation,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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
