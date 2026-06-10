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

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


def _torch_lt_2_6() -> bool:
    major, minor, *_ = (int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
    return (major, minor) < (2, 6)


def _as_int(x) -> int:
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def _block_sequence_ids_from_token_type_ids(
    token_type_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Same grouping logic as transformers Gemma3/PaliGemma image masks."""
    is_image = (token_type_ids == 1).to(device=device)
    is_previous_image = F.pad(is_image, (1, 0), value=0)[:, :-1]
    new_image_start = is_image & ~is_previous_image
    group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
    return torch.where(is_image, group_ids, -1)


def _materialize_eager_attn_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int,
    kv_offset: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask_2d: torch.Tensor | None = None,
    block_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build (B, 1, Q, K) eager mask: 0 = attend, dtype.min = mask out."""
    q_pos = torch.arange(q_length, device=device) + q_offset
    kv_pos = torch.arange(kv_length, device=device) + kv_offset
    causal = kv_pos.view(1, -1) <= q_pos.view(-1, 1)
    bool_mask = causal.view(1, 1, q_length, kv_length).expand(batch_size, 1, q_length, kv_length).clone()

    if block_sequence_ids is not None:
        seq_len = block_sequence_ids.shape[-1]
        q_clamped = q_pos.clamp(max=max(seq_len - 1, 0))
        kv_clamped = kv_pos.clamp(max=max(seq_len - 1, 0))
        q_group = block_sequence_ids[:, q_clamped]
        kv_group = block_sequence_ids[:, kv_clamped]
        q_group = torch.where(q_pos.unsqueeze(0) < seq_len, q_group, -1)
        kv_group = torch.where(kv_pos.unsqueeze(0) < seq_len, kv_group, -1)
        block_bidir = (q_group.unsqueeze(2) == kv_group.unsqueeze(1)) & (q_group.unsqueeze(2) >= 0)
        bool_mask = bool_mask | block_bidir.unsqueeze(1)

    if attention_mask_2d is not None:
        pad = attention_mask_2d.to(device=device)
        if pad.shape[-1] < kv_length:
            pad = F.pad(pad, (0, kv_length - pad.shape[-1]), value=0)
        elif pad.shape[-1] > kv_length:
            pad = pad[..., :kv_length]
        bool_mask = bool_mask & pad[:, None, None, :].to(dtype=torch.bool)

    min_dtype = torch.finfo(dtype).min
    return torch.where(bool_mask, torch.zeros((), device=device, dtype=dtype), min_dtype)


def _infer_mask_geometry(masking_utils, config, inputs_embeds, attention_mask, past_key_values, position_ids):
    layer_idx = 0
    if hasattr(masking_utils, "_preprocess_mask_arguments"):
        early_exit, attention_mask, _, q_length, kv_length, q_offset, kv_offset = (
            masking_utils._preprocess_mask_arguments(
                config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx
            )
        )
        if early_exit:
            return True, attention_mask, None, None, None, None
        return False, attention_mask, q_length, kv_length, _as_int(q_offset), _as_int(kv_offset)

    q_length = inputs_embeds.shape[1]
    if past_key_values is not None:
        q_offset = _as_int(past_key_values.get_seq_length())
        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
        kv_length, kv_offset = _as_int(kv_length), _as_int(kv_offset)
    elif attention_mask is not None:
        q_offset, kv_length, kv_offset = 0, attention_mask.shape[-1], 0
    else:
        q_offset, kv_length, kv_offset = 0, q_length, 0
    return False, attention_mask, q_length, kv_length, q_offset, kv_offset


def _create_causal_mask_torch25(masking_utils, orig_fn: Callable, **kwargs):
    """Pure-tensor causal (+ optional image block) mask; never uses vmap."""
    config = kwargs.get("config")
    inputs_embeds = kwargs.get("inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = kwargs.get("input_embeds")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values")
    position_ids = kwargs.get("position_ids")
    block_sequence_ids = kwargs.get("block_sequence_ids")

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        return attention_mask

    early_exit, attention_mask, q_length, kv_length, q_offset, kv_offset = _infer_mask_geometry(
        masking_utils, config, inputs_embeds, attention_mask, past_key_values, position_ids
    )
    if early_exit:
        return attention_mask

    batch_size = inputs_embeds.shape[0]
    attn_2d = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
    return _materialize_eager_attn_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
        attention_mask_2d=attn_2d,
        block_sequence_ids=block_sequence_ids,
    )


def _create_masks_for_generate_torch25(masking_utils, config, inputs_embeds, attention_mask, past_key_values, **kwargs):
    effective_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    position_ids = kwargs.get("position_ids")
    block_sequence_ids = kwargs.get("block_sequence_ids")
    base = dict(
        config=effective_config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        block_sequence_ids=block_sequence_ids,
    )
    if hasattr(effective_config, "layer_types"):
        return {
            layer_pattern: masking_utils.create_causal_mask(**base)
            for layer_pattern in set(effective_config.layer_types)
        }
    if getattr(effective_config, "sliding_window", None) is not None:
        return masking_utils.create_sliding_window_causal_mask(**base)
    return masking_utils.create_causal_mask(**base)


def _patch_masking_utils_for_torch25() -> None:
    """Replace transformers mask builders on torch 2.5 — bypass vmap entirely."""
    if not _torch_lt_2_6():
        return
    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        return
    if getattr(masking_utils, "_gemma3_torch25_mask_patch", False):
        return

    orig_create_causal = masking_utils.create_causal_mask
    orig_create_sliding = getattr(masking_utils, "create_sliding_window_causal_mask", None)
    orig_create_masks = masking_utils.create_masks_for_generate

    def create_causal_mask_patched(**kwargs):
        return _create_causal_mask_torch25(masking_utils, orig_create_causal, **kwargs)

    def create_sliding_window_causal_mask_patched(**kwargs):
        # Sliding layers are disabled for Gemma3 on torch 2.5; reuse full causal mask.
        return create_causal_mask_patched(**kwargs)

    def create_masks_for_generate_patched(config, inputs_embeds, attention_mask, past_key_values, **kwargs):
        kwargs.pop("or_mask_function", None)
        kwargs.pop("and_mask_function", None)
        return _create_masks_for_generate_torch25(
            masking_utils, config, inputs_embeds, attention_mask, past_key_values, **kwargs
        )

    masking_utils.create_causal_mask = create_causal_mask_patched
    if orig_create_sliding is not None:
        masking_utils.create_sliding_window_causal_mask = create_sliding_window_causal_mask_patched
    masking_utils.create_masks_for_generate = create_masks_for_generate_patched
    masking_utils._gemma3_torch25_mask_patch = True


def _patch_gemma3_mask_entrypoints() -> None:
    """Route Gemma3 image masks through block_sequence_ids + tensor mask builder."""
    if not _torch_lt_2_6():
        return
    try:
        import transformers.masking_utils as masking_utils
        from transformers.models.gemma3.modular_gemma3 import Gemma3ForConditionalGeneration
        import transformers.models.gemma3.modular_gemma3 as g3_mod
    except ImportError:
        return
    if getattr(g3_mod, "_gemma3_torch25_mask_patch", False):
        return

    def _mask_kwargs_from_token_types(config, inputs_embeds, attention_mask, past_key_values, position_ids, token_type_ids):
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if token_type_ids is not None:
            mask_kwargs["block_sequence_ids"] = _block_sequence_ids_from_token_type_ids(
                token_type_ids, inputs_embeds.device
            )
        return mask_kwargs

    @staticmethod
    def create_masks_for_generate_patched(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids=None,
        token_type_ids=None,
        is_first_iteration=False,
        **kwargs,
    ):
        mask_kwargs = _mask_kwargs_from_token_types(
            config, inputs_embeds, attention_mask, past_key_values, position_ids, token_type_ids
        )
        return masking_utils.create_masks_for_generate(**mask_kwargs)

    def create_causal_mask_mapping_patched(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids=None,
        token_type_ids=None,
        **kwargs,
    ):
        mask_kwargs = _mask_kwargs_from_token_types(
            config, inputs_embeds, attention_mask, past_key_values, position_ids, token_type_ids
        )
        return masking_utils.create_masks_for_generate(**mask_kwargs)

    Gemma3ForConditionalGeneration.create_masks_for_generate = create_masks_for_generate_patched
    g3_mod.create_causal_mask_mapping = create_causal_mask_mapping_patched
    # modeling code imports these names at module load; rebind to patched helpers.
    g3_mod.create_causal_mask = masking_utils.create_causal_mask
    g3_mod.create_masks_for_generate = masking_utils.create_masks_for_generate
    if hasattr(g3_mod, "create_sliding_window_causal_mask"):
        g3_mod.create_sliding_window_causal_mask = masking_utils.create_sliding_window_causal_mask

    try:
        import transformers.models.gemma3.modeling_gemma3 as g3_modeling
        g3_modeling.create_causal_mask = masking_utils.create_causal_mask
        g3_modeling.create_masks_for_generate = masking_utils.create_masks_for_generate
        if hasattr(g3_modeling, "create_sliding_window_causal_mask"):
            g3_modeling.create_sliding_window_causal_mask = masking_utils.create_sliding_window_causal_mask
        if hasattr(g3_modeling, "create_causal_mask_mapping"):
            g3_modeling.create_causal_mask_mapping = create_causal_mask_mapping_patched
    except ImportError:
        pass

    g3_mod._gemma3_torch25_mask_patch = True


def _disable_sliding_attention(model) -> None:
    """Force full attention only on torch 2.5."""
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

    On torch<2.6, replaces transformers mask construction with pure tensor ops
    so Gemma3 multimodal generation does not require torch 2.6 vmap support.
    """
    _patch_masking_utils_for_torch25()
    _patch_gemma3_mask_entrypoints()

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
