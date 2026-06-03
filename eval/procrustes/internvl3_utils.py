"""InternVL3-8B helpers.

InternVL3-8B uses Qwen2.5-7B as the LLM backbone (28 layers, hidden 3584),
distinct from InternVL3.5-8B which uses a Qwen3 backbone (36 layers, hidden 4096).
All other plumbing — dynamic tiling, blank PIL, decoder-path probe, AutoModel +
trust_remote_code loading — is identical to the existing InternVL helpers.
"""
from __future__ import annotations

import torch

from procrustes.internvl_utils import (
    build_transform,
    dynamic_preprocess,
    load_image_pixel_values,
    make_blank_pil,
    get_decoder_layers,
)


def load_internvl3(
    model_path: str = "/hub/huggingface/models/OpenGVLab/InternVL3-8B",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    use_flash_attn: bool = False,
):
    """Load InternVL3-8B via AutoModel(trust_remote_code=True).

    Shares the `all_tied_weights_keys` monkey-patch with `load_internvl` so
    transformers 5.x's `infer_auto_device_map` does not break on the
    InternVLChatModel `_tied_weights_keys` list attribute.
    """
    from transformers import AutoModel, AutoTokenizer

    import torch.nn as nn
    if not hasattr(nn.Module, "_internvl_compat_patched"):
        orig_getattr = nn.Module.__getattr__

        def _patched_getattr(self, name):
            if name == "all_tied_weights_keys":
                tied = getattr(self, "_tied_weights_keys", None) or []
                return {k: [] for k in tied}
            return orig_getattr(self, name)

        nn.Module.__getattr__ = _patched_getattr
        nn.Module._internvl_compat_patched = True

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


__all__ = [
    "build_transform",
    "dynamic_preprocess",
    "load_image_pixel_values",
    "make_blank_pil",
    "get_decoder_layers",
    "load_internvl3",
]
