"""InternVL3.5-8B helpers — image preprocessing, prompt building, layer path.

Mirrors the reference implementation in the model's README. Self-contained so
neither extract nor generate scripts need to duplicate the logic.
"""
from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = True):
    orig_w, orig_h = image.size
    ar = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                 for i in range(1, n + 1) for j in range(1, n + 1)
                 if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target = _find_closest_aspect_ratio(ar, target_ratios, orig_w, orig_h, image_size)
    tw, th = image_size * target[0], image_size * target[1]
    blocks = target[0] * target[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = []
    for i in range(blocks):
        x0 = (i % cols) * image_size
        y0 = (i // cols) * image_size
        tiles.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image_pixel_values(image_or_path, *, max_num: int = 12, input_size: int = 448,
                             dtype=torch.bfloat16):
    """Returns pixel_values [N_tiles, 3, 448, 448]."""
    if isinstance(image_or_path, Image.Image):
        image = image_or_path
    else:
        image = Image.open(image_or_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True,
                               max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values.to(dtype)


def make_blank_pil(size: int = 448, color=(255, 255, 255)):
    return Image.new("RGB", (size, size), color=color)


def get_decoder_layers(model):
    """InternVLChatModel.language_model = Qwen3ForCausalLM.
    Layers under: model.language_model.model.layers"""
    # try standard paths in order
    for path in [
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
    ]:
        node = model
        ok = True
        for part in path:
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok:
            return list(node)
    raise AttributeError("could not locate InternVL decoder layers")


def load_internvl(model_path: str, dtype=torch.bfloat16, device_map: str = "auto",
                  use_flash_attn: bool = False):
    """Load InternVL3.5-8B via AutoModel (trust_remote_code=True).

    Transformers 5.x introduced an `all_tied_weights_keys` attribute that
    InternVLChatModel's custom code does not define. To avoid breaking on
    `device_map='auto'`, we monkey-patch the attribute before loading.
    """
    from transformers import AutoModel, AutoTokenizer

    # transformers 5.x compat shim: InternVLChatModel uses old `_tied_weights_keys`
    # only; new infer_auto_device_map expects `all_tied_weights_keys`. Patch via
    # a property on torch.nn.Module to fallback when missing.
    import torch.nn as nn
    if not hasattr(nn.Module, "_internvl_compat_patched"):
        orig_getattr = nn.Module.__getattr__

        def _patched_getattr(self, name):
            if name == "all_tied_weights_keys":
                tied = getattr(self, "_tied_weights_keys", None) or []
                # transformers 5.x calls .keys() on this — return dict-like
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer
