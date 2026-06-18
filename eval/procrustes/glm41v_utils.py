"""GLM-4.1V-9B-Thinking helpers.

GLM-4.1V uses Glm4vForConditionalGeneration (40-layer GLM-4 text backbone at
`model.model.language_model.layers`) with a Glm4vVisionModel tower.

Official inference (zai-org/GLM-V inference/trans_infer_cli.py):
    messages = [{"role": "user", "content": [
        {"type": "image", "url": "/path/to.jpg"},
        {"type": "text", "text": question},
    ]}]
    inputs = processor.apply_chat_template(..., tokenize=True, return_dict=True, ...)
    inputs.pop("token_type_ids", None)
"""
from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image


def make_blank_pil(size: int = 448, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def get_decoder_layers(model):
    """Glm4vForConditionalGeneration → model.model.language_model.layers."""
    layers = model.model.language_model.layers
    assert len(layers) == 40, f"expected 40 GLM-4.1V decoder layers, got {len(layers)}"
    return layers


def load_glm41v(
    model_path: str = "/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load GLM-4.1V-9B-Thinking via Glm4vForConditionalGeneration + AutoProcessor."""
    from transformers import AutoProcessor, Glm4vForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    ).eval()
    return model, processor


def build_glm41v_messages(question: str, image_path: str | None) -> list[dict]:
    """Build a single-turn user message for apply_chat_template."""
    content: list[dict] = []
    if image_path:
        content.append({"type": "image", "url": image_path})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


__all__ = [
    "load_glm41v",
    "make_blank_pil",
    "get_decoder_layers",
    "build_glm41v_messages",
]
