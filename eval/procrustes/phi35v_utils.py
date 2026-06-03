"""Phi-3.5-Vision-Instruct helpers.

Phi-3.5-V uses the Phi-3.5-Mini (3.8B) LLM backbone wrapped in a custom
Phi3VForCausalLM with `model.model.layers` (32 decoder layers, hidden 3072).
The processor accepts `<|image_1|>` ... `<|image_N|>` placeholders inside the
chat prompt and stitches in CLIP-ViT image features at those positions.

This module exposes the same surface as `internvl_utils` so the extraction /
baseline / generation scripts can reuse the existing fit + hook code:
    load_phi35v(), make_blank_pil(), get_decoder_layers(), build_phi35v_prompt().
"""
from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image


def make_blank_pil(size: int = 448, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def get_decoder_layers(model):
    """Phi-3.5-V is a flat CausalLM: model.model.layers."""
    layers = model.model.layers
    assert len(layers) == 32, f"expected 32 Phi-3.5-V decoder layers, got {len(layers)}"
    return layers


def load_phi35v(
    model_path: str = "/hub/huggingface/models/microsoft/Phi-3.5-vision-instruct",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load Phi-3.5-V via AutoModelForCausalLM + AutoProcessor (trust_remote_code).

    Uses `attn_implementation="eager"` by default to avoid flash-attention2
    dependency; switch to "flash_attention_2" if flash-attn is installed.
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
    # transformers >=5.x auto-routes configs with sliding_window to
    # DynamicSlidingWindowLayer cache, which the bundled Phi-3 modeling code
    # was not written against. Disable to force plain DynamicLayer cache.
    if getattr(model.config, "sliding_window", None):
        model.config.sliding_window = None
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, num_crops=4
    )
    return model, processor


def build_phi35v_prompt(processor, question: str, has_image: bool) -> str:
    """Return a chat prompt string with optional `<|image_1|>` placeholder."""
    content = (f"<|image_1|>\n{question}" if has_image else question)
    messages = [{"role": "user", "content": content}]
    return processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


__all__ = [
    "load_phi35v",
    "make_blank_pil",
    "get_decoder_layers",
    "build_phi35v_prompt",
]
