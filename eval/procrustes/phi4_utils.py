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


def _patch_dynamic_cache_compat() -> None:
    """Shim legacy KV-cache APIs expected by remote ``modeling_phi4mm.py``.

    transformers ≥ 4.50 removed ``Cache.get_usable_length`` / ``get_max_length``.
    Phi-4 remote attention still calls them during ``model.generate()``.
    """
    try:
        import transformers.cache_utils as cache_utils
    except ImportError:
        return
    if getattr(cache_utils, "_phi4_cache_compat_patch", False):
        return

    def _get_usable_length(self, seq_length=0, layer_idx=0):
        if hasattr(self, "get_seq_length"):
            try:
                return int(self.get_seq_length(layer_idx))
            except TypeError:
                return int(self.get_seq_length())
        return 0

    def _get_max_length(self):
        if hasattr(self, "get_max_cache_shape"):
            shape = self.get_max_cache_shape()
            if shape is not None:
                return shape
        return 4096

    for name in dir(cache_utils):
        cls = getattr(cache_utils, name, None)
        if not isinstance(cls, type):
            continue
        if not hasattr(cls, "get_usable_length"):
            cls.get_usable_length = _get_usable_length
        if not hasattr(cls, "get_max_length"):
            cls.get_max_length = _get_max_length

    cache_utils._phi4_cache_compat_patch = True


def ensure_phi4_transformers_compat() -> None:
    """Apply all transformers-side shims needed for Phi-4 remote code."""
    _patch_dynamic_cache_compat()


def _patch_phi4mm_forward(mod) -> None:
    """Guard Phi4MMForCausalLM.forward when num_logits_to_keep is None."""
    cls = getattr(mod, "Phi4MMForCausalLM", None)
    if cls is None or getattr(cls, "_phi4_forward_patched", False):
        return
    orig_forward = cls.forward

    def forward(self, *args, num_logits_to_keep=None, **kwargs):
        if num_logits_to_keep is None:
            num_logits_to_keep = 1
        return orig_forward(self, *args, num_logits_to_keep=num_logits_to_keep, **kwargs)

    cls.forward = forward
    cls._phi4_forward_patched = True


def _patch_phi4mm_cls(cls) -> bool:
    """Add prepare_inputs_for_generation stub if missing on a real Python class."""
    if not isinstance(cls, type):
        # torch._classes ScriptClass namespace — cannot be patched.
        return False
    if hasattr(cls, "prepare_inputs_for_generation"):
        return True  # already present

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    cls.prepare_inputs_for_generation = prepare_inputs_for_generation
    return True


def _preload_and_patch_phi4_remote(model_path: str) -> None:
    """Import remote modeling_phi4mm and patch Phi4MMModel before weight load."""
    import sys

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM", "modeling_phi4mm.Phi4MMForCausalLM")
    causal_cls = get_class_from_dynamic_module(class_ref, model_path)

    # --- Locate Phi4MMModel via the returned class's own module ---------------
    # get_class_from_dynamic_module returns a real Python class whose __module__
    # points to the dynamic transformers_modules entry in sys.modules.  Going
    # through that module avoids iterating sys.modules and accidentally hitting
    # torch._classes ScriptClass namespaces (whose __getattr__ raises
    # RuntimeError instead of AttributeError, breaking hasattr()).
    patched = False
    mod = sys.modules.get(causal_cls.__module__) if causal_cls is not None else None
    if mod is not None and hasattr(mod, "Phi4MMModel"):
        patched = _patch_phi4mm_cls(mod.Phi4MMModel)
    if mod is not None:
        _patch_phi4mm_forward(mod)

    if not patched:
        raise RuntimeError(
            "Failed to patch Phi4MMModel.prepare_inputs_for_generation for PEFT. "
            "Try: rm -rf ~/.cache/huggingface/modules/transformers_modules/Phi_hyphen_4_hyphen_multimodal_hyphen_instruct"
        )


def _load_phi4_remote(
    model_path: str,
    dtype,
    device_map: str,
    attn_implementation: str,
):
    from transformers import AutoModelForCausalLM

    _preload_and_patch_phi4_remote(model_path)
    return _load_pretrained(
        AutoModelForCausalLM,
        model_path,
        dtype,
        device_map,
        attn_implementation,
        trust_remote_code=True,
        _attn_implementation=attn_implementation,
    )


def load_phi4(
    model_path: str = "/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct",
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load Phi-4-multimodal via remote modeling_phi4mm + AutoProcessor.

    microsoft/Phi-4-multimodal-instruct uses ``model_type=phi4mm`` (remote code),
    not the in-tree ``phi4_multimodal`` class. We patch ``Phi4MMModel`` for PEFT
    compatibility before loading weights.

    Defaults to eager attention (no flash-attn requirement). Set
    attn_implementation="flash_attention_2" on Ampere+ if flash-attn is installed.
    """
    from transformers import AutoConfig, AutoProcessor

    ensure_phi4_transformers_compat()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    # Only HF repos published as phi4_multimodal can use the in-tree class.
    if model_type == "phi4_multimodal":
        try:
            from transformers import Phi4MultimodalForCausalLM

            model = _load_pretrained(
                Phi4MultimodalForCausalLM,
                model_path,
                dtype,
                device_map,
                attn_implementation,
            )
            model.config.use_cache = False
            return model, processor
        except Exception:
            pass

    model = _load_phi4_remote(model_path, dtype, device_map, attn_implementation)
    model.config.use_cache = False
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
    "ensure_phi4_transformers_compat",
]
