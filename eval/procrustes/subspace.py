"""Per-layer subspace fitting + orthogonal Procrustes solver.

Two subspace modes:

  split    : independent top-k SVD on H_c and H_t. B_c and B_t span generally
             different subspaces. This is the literal recipe in the design
             doc; geometric interpretation of `h - p_c + p_t` is unclear here
             so we expose it mostly for ablation.

  shared   : top-k SVD on the stacked matrix [X_c; X_t] (both centered or
             both raw, depending on `centering_mode`). B_c == B_t == B and Q
             is a clean within-subspace rotation.

Two centering modes:

  none                       : raw (uncentered) SVD. Per the doc's No-Center
                               variant. WARNING: in deep LLaVA layers with
                               ||mu|| ~ 100 this nearly always sets B[:,0]
                               parallel to mu and Q ~ identity on the mean
                               axis -- effectively wasting one rank.

  mean_preserving_centered   : centered SVD + remember mu_t - mu_c so the
                               intervention can re-add lambda * mean_shift.
"""
from __future__ import annotations

from typing import Any


def _as_tensor(value: Any, *, device=None, dtype=None):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (list, tuple)) and value and all(
        isinstance(x, torch.Tensor) for x in value
    ):
        tensor = torch.stack([x.detach().cpu() for x in value], dim=0)
    else:
        tensor = torch.tensor(value)
    if dtype is None and not tensor.is_floating_point():
        dtype = torch.float32
    return tensor.to(device=device, dtype=dtype or tensor.dtype)


def topk_svd_basis(X, k: int):
    """Return [d, k] right-singular-vectors of X = U S Vh (top-k by sigma).

    Pass raw H for `centering_mode=none`, or H - mean(H) for centered.
    """
    import torch

    X = _as_tensor(X, dtype=torch.float32)
    if X.ndim != 2:
        raise ValueError(f"expected [N, d], got shape {tuple(X.shape)}")
    if X.shape[0] == 0:
        raise ValueError("empty rows; cannot fit subspace")
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    k = min(k, Vh.shape[0])
    return Vh[:k].T.contiguous()


def shared_topk_svd_basis(X_c, X_t, k: int):
    """Top-k SVD on the row-stacked matrix [X_c; X_t]. Returns [d, k] basis B.

    Use this when you want B_c == B_t == B and Q to be a clean
    within-subspace rotation.
    """
    import torch

    X_c = _as_tensor(X_c, dtype=torch.float32)
    X_t = _as_tensor(X_t, dtype=torch.float32)
    if X_c.shape[1] != X_t.shape[1]:
        raise ValueError(f"d mismatch: {X_c.shape[1]} vs {X_t.shape[1]}")
    return topk_svd_basis(torch.cat([X_c, X_t], dim=0), k)


def solve_orthogonal_procrustes(Z_c, Z_t, proper_rotation: bool = True):
    """Solve min_Q ||Z_c Q - Z_t||_F^2 s.t. Q^T Q = I.

    Returns Q in [k, k]. With proper_rotation=True, det(Q) = +1
    (excludes reflections).
    """
    import torch

    Z_c = _as_tensor(Z_c, dtype=torch.float32)
    Z_t = _as_tensor(Z_t, dtype=torch.float32)
    if Z_c.shape != Z_t.shape:
        raise ValueError(f"Z_c {tuple(Z_c.shape)} vs Z_t {tuple(Z_t.shape)} mismatch")
    M = Z_c.T @ Z_t
    P, _, Rt = torch.linalg.svd(M, full_matrices=False)
    if not proper_rotation:
        return (P @ Rt).contiguous()
    D = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    D[-1, -1] = torch.det(P @ Rt).sign()
    return (P @ D @ Rt).contiguous()


def fit_layer(
    H_t,
    H_c,
    *,
    k: int,
    centering_mode: str = "mean_preserving_centered",
    subspace_mode: str = "shared",
    proper_rotation: bool = True,
) -> dict:
    """Fit one layer's Procrustes parameters from paired hidden states.

    Returns dict with B_t, B_c, Q, mu_t, mu_c, mean_shift, plus diagnostics.
    """
    import torch

    if centering_mode not in ("none", "mean_preserving_centered"):
        raise ValueError(f"unknown centering_mode {centering_mode!r}")
    if subspace_mode not in ("split", "shared"):
        raise ValueError(f"unknown subspace_mode {subspace_mode!r}")

    H_t = _as_tensor(H_t, dtype=torch.float32)
    H_c = _as_tensor(H_c, dtype=torch.float32)
    if H_t.shape != H_c.shape:
        raise ValueError(f"H_t {tuple(H_t.shape)} vs H_c {tuple(H_c.shape)} mismatch")
    d = H_t.shape[1]

    if centering_mode == "none":
        mu_t = torch.zeros(d, dtype=torch.float32)
        mu_c = torch.zeros(d, dtype=torch.float32)
        X_t, X_c = H_t, H_c
    else:
        mu_t = H_t.mean(dim=0)
        mu_c = H_c.mean(dim=0)
        X_t = H_t - mu_t
        X_c = H_c - mu_c
    mean_shift = mu_t - mu_c

    if subspace_mode == "split":
        B_t = topk_svd_basis(X_t, k)
        B_c = topk_svd_basis(X_c, k)
    else:  # shared
        B = shared_topk_svd_basis(X_c, X_t, k)
        B_t = B
        B_c = B

    Z_t = X_t @ B_t
    Z_c = X_c @ B_c
    Q = solve_orthogonal_procrustes(Z_c, Z_t, proper_rotation=proper_rotation)

    pre = (Z_c - Z_t).norm() / (Z_t.norm() + 1e-8)
    post = (Z_c @ Q - Z_t).norm() / (Z_t.norm() + 1e-8)
    det_q = torch.det(Q)

    # If subspace is shared, B_c is exactly B_t -- the orientation diagnostic
    # is meaningful only when split.
    if subspace_mode == "split":
        # principal-angle cosines between the column spans of B_c and B_t
        # (numerically: singular values of B_c.T @ B_t are cos(phi_i)).
        sing = torch.linalg.svdvals(B_c.T @ B_t).clamp(0.0, 1.0)
        principal_cos = sing.tolist()
    else:
        principal_cos = [1.0] * k

    return {
        "B_t": B_t.cpu(),
        "B_c": B_c.cpu(),
        "Q": Q.cpu(),
        "mu_t": mu_t.cpu(),
        "mu_c": mu_c.cpu(),
        "mean_shift": mean_shift.cpu(),
        "k": int(k),
        "centering_mode": centering_mode,
        "subspace_mode": subspace_mode,
        "proper_rotation": bool(proper_rotation),
        "diagnostics": {
            "pre_error": float(pre),
            "post_error": float(post),
            "det_Q": float(det_q),
            "principal_angle_cos": principal_cos,
        },
    }


def fit_all_layers(
    H_t_by_layer: dict,
    H_c_by_layer: dict,
    *,
    k: int,
    centering_mode: str = "mean_preserving_centered",
    subspace_mode: str = "shared",
    proper_rotation: bool = True,
) -> dict[int, dict]:
    layers = sorted(set(H_t_by_layer) & set(H_c_by_layer))
    return {
        int(l): fit_layer(
            H_t_by_layer[l],
            H_c_by_layer[l],
            k=k,
            centering_mode=centering_mode,
            subspace_mode=subspace_mode,
            proper_rotation=proper_rotation,
        )
        for l in layers
    }


def apply_layer(
    h,
    params: dict,
    *,
    alpha: float,
    lambda_mean: float = 0.0,
    rotation_scale: float = 1.0,
    mean_shift_mode: str = "full",
    refusal_dir: object | None = None,
    G_orth: object | None = None,
    mean_shift_gate_threshold: float | None = None,
    mean_shift_gate_temp: float = 0.05,
):
    """Apply one layer's Procrustes intervention to a single hidden tensor.

    h: tensor of shape [..., d] (typically [B, d] or [B, T, d] with the caller
       slicing the last token before passing in).

    mean_shift_mode: how to compute the mean-shift contribution. Options:
      - 'full' (default, backward compatible): use lambda_mean * (mu_t - mu_c).
        A scalar lambda_mean is applied uniformly to the full mean-shift vector.
      - 'projected_rotation': project (mu_t - mu_c) onto the per-sample
        rotation correction direction delta_rot = (-p_c + p_t).
      - 'refusal_projected': project (mu_t - mu_c) onto the per-layer refusal
        direction r̂^l (unit vector from refusal probe). Mean-shift contribution
        becomes ⟨μ_t-μ_c, r̂⟩ · r̂. Requires refusal_dir to be passed in.

    refusal_dir: optional tensor[d] (or dict {"r": tensor[d]}) — unit refusal
       direction for the current layer, used by mean_shift_mode='refusal_projected'.

    G_orth: optional [d, k_g] orthonormal basis of a "benign-task subspace"
       to orthogonalize the *correction* against.
    """
    import torch

    if mean_shift_mode not in ("full", "projected_rotation", "refusal_projected"):
        raise ValueError(f"unknown mean_shift_mode {mean_shift_mode!r}")

    centering_mode = params["centering_mode"]
    B_t = params["B_t"].to(device=h.device, dtype=h.dtype)
    B_c = params["B_c"].to(device=h.device, dtype=h.dtype)
    Q = params["Q"].to(device=h.device, dtype=h.dtype)
    mu_c = params["mu_c"].to(device=h.device, dtype=h.dtype)
    mean_shift = params["mean_shift"].to(device=h.device, dtype=h.dtype)

    if centering_mode == "none":
        x = h
        mean_term = torch.zeros_like(h)
    else:
        x = h - mu_c
        mean_term = None  # computed below

    z = x @ B_c
    p_c = z @ B_c.T
    z_rot = z @ Q
    p_t = z_rot @ B_t.T
    delta_rot = -p_c + p_t  # rotation correction (per-sample)

    if centering_mode != "none":
        if mean_shift_mode == "full":
            mean_term = lambda_mean * mean_shift
        elif mean_shift_mode == "projected_rotation":
            inner = (mean_shift * delta_rot).sum(dim=-1, keepdim=True)
            norm_sq = (delta_rot * delta_rot).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            beta = inner / norm_sq
            mean_term = beta * delta_rot
        else:  # refusal_projected
            if refusal_dir is None:
                raise ValueError("mean_shift_mode='refusal_projected' requires refusal_dir")
            r = refusal_dir["r"] if isinstance(refusal_dir, dict) else refusal_dir
            r = r if hasattr(r, "device") else torch.as_tensor(r)
            r = r.to(device=h.device, dtype=h.dtype)
            # Make sure r is unit
            r = r / (r.norm() + 1e-8)
            # <mean_shift, r> is a scalar; multiply by r and lambda_mean
            coeff = (mean_shift * r).sum()  # scalar
            # Broadcast to match h shape
            mean_term = lambda_mean * coeff * r  # shape [d]
            mean_term = mean_term.expand_as(h) if mean_term.dim() < h.dim() else mean_term

    if mean_shift_gate_threshold is not None and centering_mode != "none":
        num = p_c.pow(2).sum(dim=-1, keepdim=True)
        den = x.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ratio = num / den
        gate = torch.sigmoid((ratio - mean_shift_gate_threshold) / mean_shift_gate_temp)
        mean_term = gate * mean_term

    rotation_term = rotation_scale * delta_rot
    correction = rotation_term + mean_term

    if G_orth is not None:
        G_raw = G_orth.get("basis") if isinstance(G_orth, dict) else G_orth
        G = G_raw if hasattr(G_raw, "device") else torch.as_tensor(G_raw)
        G = G.to(device=h.device, dtype=h.dtype)
        coeff = correction @ G
        correction = correction - coeff @ G.T

    return h + alpha * correction


def fit_benign_subspace(H_benign_by_layer: dict, *, k: int, centered: bool = True) -> dict[int, object]:
    """Fit per-layer top-k benign-task subspace from text-only hidden states
    on benign queries. Returns dict {layer: tensor[d, k]} with orthonormal cols.

    Use this with apply_layer(..., G_orth=basis[layer]) to orthogonalize the
    Procrustes correction against the benign-task subspace.
    """
    import torch

    out: dict[int, object] = {}
    for layer, H in H_benign_by_layer.items():
        H_t = _as_tensor(H, dtype=torch.float32)
        if centered:
            H_t = H_t - H_t.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(H_t, full_matrices=False)
        # Vh: [min(N,d), d]; top-k right singular vectors are PCA basis.
        k_eff = min(k, Vh.shape[0])
        out[int(layer)] = Vh[:k_eff].T.contiguous().cpu()
    return out


def fit_benign_subspace_with_mean(
    H_benign_by_layer: dict,
    *,
    k: int,
    centered: bool = True,
) -> dict[int, dict[str, object]]:
    """Fit utility/benign subspace and retain the centering mean.

    The returned object is useful for runtime guards that compare a current
    hidden state against the utility manifold. It is also accepted by
    `apply_layer(..., G_orth=...)`, which reads the `basis` field.
    """
    import torch

    out: dict[int, dict[str, object]] = {}
    for layer, H in H_benign_by_layer.items():
        H_t = _as_tensor(H, dtype=torch.float32)
        mean = H_t.mean(dim=0)
        X = H_t - mean if centered else H_t
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        k_eff = min(k, Vh.shape[0])
        out[int(layer)] = {
            "basis": Vh[:k_eff].T.contiguous().cpu(),
            "mean": mean.contiguous().cpu(),
            "centered": bool(centered),
        }
    return out
