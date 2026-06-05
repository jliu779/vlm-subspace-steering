"""Gemma-3-4B-Instruct helpers.

Gemma-3-4B-IT is a Gemma3ForConditionalGeneration model (Gemma-3 4B LLM
backbone, 34 decoder layers, hidden 2560) with a SigLIP vision tower wrapped
under `model.vision_tower`. Decoder layers live at
`model.model.language_model.layers`. Native in transformers 5.7+.

This module exposes the same surface as `phi35v_utils` / `internvl3_utils`
so the existing fit + hook scripts can reuse it:
    load_gemma3(), make_blank_pil(), get_decoder_layers(), build_gemma3_prompt().
"""
from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image


def make_blank_pil(size: int = 896, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def get_decoder_layers(model):
    """Gemma3ForConditionalGeneration → model.model.language_model.layers."""
    layers = model.model.language_model.layers
    assert len(layers) == 34, f"expected 34 Gemma-3-4B decoder layers, got {len(layers)}"
    return layers


def load_gemma3(
    model_path: str = "/hub/huggingface/models/google/gemma-3-4b-it",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load Gemma-3-4B-IT via Gemma3ForConditionalGeneration + AutoProcessor.

    `attn_implementation="eager"` avoids flash-attn dependency. Defensive
    sliding_window=None on the text_config to mirror the Phi-3.5-V fix in case
    transformers' cache router mis-routes Gemma-3.
    """
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_gemma3_prompt(processor, question: str, has_image: bool) -> str:
    """Return a chat-template prompt string, optionally including an image slot."""
    if has_image:
        content = [{"type": "image"}, {"type": "text", "text": question}]
    else:
        content = [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


__all__ = [
    "load_gemma3",
    "make_blank_pil",
    "get_decoder_layers",
    "build_gemma3_prompt",
]
