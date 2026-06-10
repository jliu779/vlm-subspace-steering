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


def _torch_lt_2_6() -> bool:
    major, minor, *_ = (int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
    return (major, minor) < (2, 6)


def _patch_masking_utils_for_torch25() -> None:
    """Allow Gemma3 multimodal masks on torch 2.5.x.

    transformers>=5.x gates `or_mask_function` / `and_mask_function` behind
    torch>=2.6 even though eager attention on 2.5 can materialize the same 4D
    masks via vmap. Gemma3 always hits this path for image token_type_ids.
    """
    if not _torch_lt_2_6():
        return
    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        return
    if getattr(masking_utils, "_gemma3_torch25_mask_patch", False):
        return
    masking_utils._is_torch_greater_or_equal_than_2_6 = True
    masking_utils._gemma3_torch25_mask_patch = True


def _disable_sliding_attention(model) -> None:
    """Force full attention only — sliding-window masks share the same torch gate."""
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is None:
        return
    n = getattr(text_cfg, "num_hidden_layers", 0)
    if getattr(text_cfg, "sliding_window", None) is not None:
        text_cfg.sliding_window = None
    if getattr(text_cfg, "layer_types", None):
        text_cfg.layer_types = ["full_attention"] * n

    try:
        layers = model.model.language_model.layers
    except AttributeError:
        return
    for layer in layers:
        if hasattr(layer, "layer_type"):
            layer.layer_type = "full_attention"
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "is_sliding"):
            attn.is_sliding = False
        if hasattr(attn, "sliding_window"):
            attn.sliding_window = None


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

    `attn_implementation="eager"` avoids flash-attn dependency. On torch<2.6 we
    patch transformers masking_utils and disable sliding-window layers so
    multimodal generation does not require torch 2.6 mask combinators.
    """
    _patch_masking_utils_for_torch25()

    from transformers import Gemma3ForConditionalGeneration, AutoProcessor

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    ).eval()
    if _torch_lt_2_6():
        _disable_sliding_attention(model)
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
