from __future__ import annotations

import random
from pathlib import Path


def load_image(path: str | Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def make_blank_image_like(image, color: tuple[int, int, int] = (255, 255, 255)):
    from PIL import Image

    return Image.new("RGB", image.size, color=color)


def make_noise_image_like(image, seed: int = 0):
    from PIL import Image

    rng = random.Random(seed)
    w, h = image.size
    data = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(w * h)]
    out = Image.new("RGB", (w, h))
    out.putdata(data)
    return out
