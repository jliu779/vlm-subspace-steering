from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any


def _as_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    rows = [[float(x) for x in row] for row in value]
    if not rows:
        raise ValueError("cannot fit vector from empty deltas")
    return rows


def _mean(rows: list[list[float]]) -> list[float]:
    return [sum(col) / len(rows) for col in zip(*rows)]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _unit(vec: list[float]) -> list[float]:
    n = _norm(vec)
    return [x / n for x in vec] if n else [0.0 for _ in vec]


def _first_pc(rows: list[list[float]]) -> tuple[list[float], float]:
    """Return (unit_pc, sigma) where sigma is the std of the projection on PC1."""
    try:
        import numpy as np
        from sklearn.decomposition import PCA

        arr = np.asarray(rows, dtype="float32")
        pca = PCA(n_components=1, svd_solver="randomized", random_state=0).fit(arr)
        comp = pca.components_[0]
        sigma = float(np.sqrt(pca.explained_variance_[0]))
        return comp.tolist(), sigma
    except Exception:
        mean = _mean(rows)
        centered = [[x - m for x, m in zip(row, mean)] for row in rows]
        dim = len(rows[0])
        vec = _unit(mean)
        if not any(vec):
            vec = [1.0] + [0.0] * (dim - 1)
        for _ in range(25):
            nxt = []
            for j in range(dim):
                nxt.append(sum(row[j] * sum(a * b for a, b in zip(row, vec)) for row in centered))
            vec = _unit(nxt)
            if not any(vec):
                break
        n = len(rows)
        denom = max(n - 1, 1)
        var = sum(sum(a * b for a, b in zip(row, vec)) ** 2 for row in centered) / denom
        sigma = var ** 0.5
        return vec, sigma


def fit_layer_vectors(
    deltas_by_layer: dict[int, Any],
    pca_scale: str = "pc_scaled",
) -> tuple[dict[int, list[float]], dict[int, float]]:
    """Fit one steering vector per layer from precomputed h_t - h_c deltas.

    Returns (vectors, sigmas) where sigmas[layer] is the projection std on PC1
    for that layer. sigmas are useful for per-layer norm caps in sample-level
    CMRM (cap each sample delta to factor * sigma_1[layer] so individual
    samples can't push the residual stream off-distribution).

    pca_scale options:
      unit_pc             : first principal component, unit norm (paper-literal but tiny in absolute scale)
      pc_scaled           : first principal component scaled by its projection std (matches the magnitude
                            of the actual h_t - h_c gap; recommended default with alpha=1.0)
      mean_abs_projection : first principal component scaled by mean absolute projection
      mean_delta          : raw mean of (h_t - h_c) across the anchor set, no PCA
    """
    vectors: dict[int, list[float]] = {}
    sigmas: dict[int, float] = {}
    for layer, raw_rows in deltas_by_layer.items():
        rows = _as_rows(raw_rows)
        # Always compute sigma on PC1 even when pca_scale doesn't use it,
        # so the returned dict is sufficient to drive per-layer caps later.
        _, sigma = _first_pc(rows)
        sigmas[int(layer)] = float(sigma)
        if pca_scale == "mean_delta":
            vector = _mean(rows)
        else:
            pc, _ = _first_pc(rows)
            pc = _unit(pc)
            if pca_scale == "unit_pc":
                vector = pc
            elif pca_scale == "pc_scaled":
                vector = [sigma * x for x in pc]
            elif pca_scale == "mean_abs_projection":
                center = _mean(rows)
                scale = sum(abs(sum((x - m) * p for x, m, p in zip(row, center, pc))) for row in rows) / len(rows)
                vector = [scale * x for x in pc]
            else:
                raise ValueError(f"unknown pca_scale {pca_scale!r}")
        vectors[int(layer)] = [float(x) for x in vector]
    return vectors, sigmas


def save_vectors(
    vectors: dict[int, Any],
    meta: dict[str, Any],
    out: str | Path,
    sigmas: dict[int, float] | None = None,
) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vectors": vectors, "meta": meta}
    if sigmas is not None:
        payload["sigmas"] = sigmas
    try:
        import torch

        torch.save(payload, out)
    except Exception:
        with out.open("wb") as f:
            pickle.dump(payload, f)


def load_vectors(path: str | Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
    payload["vectors"] = {int(k): v for k, v in payload["vectors"].items()}
    if "sigmas" in payload and payload["sigmas"] is not None:
        payload["sigmas"] = {int(k): float(v) for k, v in payload["sigmas"].items()}
    return payload


def save_meta(meta: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
