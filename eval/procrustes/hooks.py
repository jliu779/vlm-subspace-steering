"""Forward-hook manager that applies per-layer Procrustes rotation to the
last input token. Same prefill_only / all_steps gating semantics as the
CMRM hook (defaulting to prefill_only because all_steps collapses LLaVA
generation into degenerate-token loops).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .subspace import apply_layer


def _hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def _basis_and_mean(subspace, *, device, dtype):
    import torch

    if isinstance(subspace, dict):
        basis = subspace["basis"]
        mean = subspace.get("mean")
    else:
        basis = subspace
        mean = None
    basis = basis if hasattr(basis, "to") else torch.as_tensor(basis)
    basis = basis.to(device=device, dtype=dtype)
    if mean is not None:
        mean = mean if hasattr(mean, "to") else torch.as_tensor(mean)
        mean = mean.to(device=device, dtype=dtype)
    return basis, mean


class ProcrustesHookManager:
    def __init__(
        self,
        layers: list[Any],
        params_by_layer: Mapping[int, dict],
        target_layers: list[int],
        *,
        alpha: float,
        lambda_mean: float = 0.0,
        rotation_scale: float = 1.0,
        mean_shift_mode: str = "full",
        refusal_dir_by_layer: Mapping[int, Any] | None = None,
        hook_scope: str = "prefill_only",
        decode_k_steps: int = 0,
        alpha_by_layer: Mapping[int, float] | None = None,
        lambda_by_layer: Mapping[int, float] | None = None,
        alpha_scale: float = 1.0,
        trust_region_rho: float | None = None,
        residual_vectors_by_layer: Mapping[int, Any] | None = None,
        residual_alpha: float = 0.0,
        residual_sign: int = 1,
        benign_subspace_by_layer: Mapping[int, Any] | None = None,
        utility_guard_by_layer: Mapping[int, Any] | None = None,
        utility_guard_threshold: float | None = None,
        utility_guard_scale: float = 0.0,
        lda_guard_by_layer: Mapping[int, Any] | None = None,
        lda_guard_scale: float = 0.0,
        mean_shift_gate_threshold: float | None = None,
        mean_shift_gate_temp: float = 0.05,
    ) -> None:
        if hook_scope not in {"prefill_only", "all_steps", "prefill_plus_decode_k"}:
            raise ValueError(f"unknown hook_scope {hook_scope!r}")
        if decode_k_steps < 0:
            raise ValueError("decode_k_steps must be >= 0")
        self.layers = layers
        self.params_by_layer = params_by_layer
        self.target_layers = [int(l) for l in target_layers]
        self.alpha = float(alpha)
        self.lambda_mean = float(lambda_mean)
        self.rotation_scale = float(rotation_scale)
        self.mean_shift_mode = str(mean_shift_mode)
        self.refusal_dir_by_layer = refusal_dir_by_layer or {}
        self.hook_scope = hook_scope
        self.decode_k_steps = int(decode_k_steps)
        self.alpha_by_layer = {int(k): float(v) for k, v in (alpha_by_layer or {}).items()}
        self.lambda_by_layer = {int(k): float(v) for k, v in (lambda_by_layer or {}).items()}
        self.alpha_scale = float(alpha_scale)
        self.trust_region_rho = (
            None if trust_region_rho is None else float(trust_region_rho)
        )
        self.residual_vectors_by_layer = residual_vectors_by_layer or {}
        self.residual_alpha = float(residual_alpha)
        self.residual_sign = 1 if int(residual_sign) >= 0 else -1
        # Optional Variant A: orthogonalize the per-layer correction against
        # this benign-task subspace before adding to h. Empirically the
        # modality-shift correction has 99.7% energy in G^perp, so this is
        # essentially free for safety while preserving utility.
        self.benign_subspace_by_layer = benign_subspace_by_layer or {}
        self.utility_guard_by_layer = utility_guard_by_layer or {}
        self.utility_guard_threshold = (
            None if utility_guard_threshold is None else float(utility_guard_threshold)
        )
        self.utility_guard_scale = float(utility_guard_scale)
        # LDA-based discriminative guard: score(h) = h @ w + b
        # score > 0 → ScienceQA-like → scale delta by lda_guard_scale
        # score < 0 → VLSafe-like   → full correction (scale = 1.0)
        self.lda_guard_by_layer = lda_guard_by_layer or {}
        self.lda_guard_scale = float(lda_guard_scale)
        # Projection-gated mean shift: scale the global mean-shift term by a
        # sigmoid of the in-subspace projection ratio. When set, benign-like
        # inputs (low ratio) receive no mean shift; harmful-like inputs (high
        # ratio) get the full shift. Provides geometric gating with zero
        # keyword dependence.
        self.mean_shift_gate_threshold = (
            None if mean_shift_gate_threshold is None else float(mean_shift_gate_threshold)
        )
        self.mean_shift_gate_temp = float(mean_shift_gate_temp)
        self.handles = []
        self._decode_step_count = 0
        self._decode_counter_layer = (
            min(self.target_layers) if self.target_layers else 0
        )

    def _should_apply(self, seq_len: int | None, layer_id: int) -> bool:
        if self.hook_scope == "all_steps":
            return True
        if self.hook_scope == "prefill_only":
            return seq_len != 1
        # prefill_plus_decode_k:
        if seq_len != 1:
            return True
        if layer_id == self._decode_counter_layer:
            self._decode_step_count += 1
        return self._decode_step_count <= self.decode_k_steps

    def _clip_delta_by_rho(self, hidden_last, delta):
        if self.trust_region_rho is None or self.trust_region_rho <= 0:
            return delta
        import torch

        rho = float(self.trust_region_rho)
        eps = 1e-8
        h_norm = torch.norm(hidden_last, dim=-1, keepdim=True)
        d_norm = torch.norm(delta, dim=-1, keepdim=True)
        max_norm = rho * h_norm
        scale = torch.where(
            d_norm > max_norm,
            max_norm / (d_norm + eps),
            torch.ones_like(d_norm),
        )
        return delta * scale

    def _apply_lda_guard(self, layer_id: int, hidden_last, delta):
        """Discriminative guard: suppress delta when LDA scores as ScienceQA-like."""
        lda = self.lda_guard_by_layer.get(layer_id)
        if lda is None:
            return delta
        import torch

        w = lda["w"].to(device=hidden_last.device, dtype=hidden_last.dtype)
        b = lda["b"].to(device=hidden_last.device, dtype=hidden_last.dtype)
        # hidden_last: (batch, 1, dim) → score per sample
        score = (hidden_last @ w).squeeze(-1) + b  # (batch, 1)
        # score > 0 → ScienceQA-like → scale by lda_guard_scale
        # score < 0 → VLSafe-like   → scale by 1.0
        scale = torch.where(
            score > 0,
            torch.full_like(score, self.lda_guard_scale),
            torch.ones_like(score),
        ).unsqueeze(-1)  # (batch, 1, 1)
        return delta * scale

    def _apply_utility_guard(self, layer_id: int, hidden_last, delta):
        if self.utility_guard_threshold is None:
            return delta
        subspace = self.utility_guard_by_layer.get(layer_id)
        if subspace is None:
            return delta
        import torch

        G, mean = _basis_and_mean(
            subspace, device=hidden_last.device, dtype=hidden_last.dtype,
        )
        x = hidden_last if mean is None else hidden_last - mean.view(1, 1, -1)
        proj = (x @ G) @ G.T
        ratio = proj.norm(dim=-1, keepdim=True).square() / (
            x.norm(dim=-1, keepdim=True).square() + 1e-8
        )
        scale = torch.where(
            ratio >= self.utility_guard_threshold,
            torch.full_like(ratio, self.utility_guard_scale),
            torch.ones_like(ratio),
        )
        return delta * scale

    def __enter__(self):
        for layer_id in self.target_layers:
            params = self.params_by_layer[layer_id]
            G = self.benign_subspace_by_layer.get(layer_id)
            utility_guard = self.utility_guard_by_layer.get(layer_id)
            lda_guard = self.lda_guard_by_layer.get(layer_id)
            residual_vec = self.residual_vectors_by_layer.get(layer_id)
            alpha_layer = self.alpha_by_layer.get(layer_id, self.alpha)
            lambda_layer = self.lambda_by_layer.get(layer_id, self.lambda_mean)

            def hook(
                _module,
                _inputs,
                output,
                params=params,
                G=G,
                utility_guard=utility_guard,
                lda_guard=lda_guard,
                residual_vec=residual_vec,
                alpha_layer=alpha_layer,
                lambda_layer=lambda_layer,
                layer_id=layer_id,
            ):
                hidden = _hidden_from_output(output)
                shape = getattr(hidden, "shape", None)
                if shape is not None and len(shape) >= 2:
                    seq_len = shape[1]
                else:
                    seq_len = None
                if not self._should_apply(seq_len, layer_id):
                    return output
                last = hidden[:, -1:, :]
                edited_last_base = apply_layer(
                    last, params,
                    alpha=alpha_layer * self.alpha_scale,
                    lambda_mean=lambda_layer,
                    rotation_scale=self.rotation_scale,
                    mean_shift_mode=self.mean_shift_mode,
                    refusal_dir=self.refusal_dir_by_layer.get(layer_id),
                    G_orth=G,
                    mean_shift_gate_threshold=self.mean_shift_gate_threshold,
                    mean_shift_gate_temp=self.mean_shift_gate_temp,
                )
                delta = edited_last_base - last
                if residual_vec is not None and self.residual_alpha > 0:
                    import torch

                    vec = (
                        residual_vec
                        if hasattr(residual_vec, "to")
                        else torch.tensor(residual_vec)
                    )
                    vec = vec.to(device=last.device, dtype=last.dtype)
                    delta = delta + (
                        self.residual_sign
                        * self.residual_alpha
                        * self.alpha_scale
                        * vec.view(1, 1, -1)
                    )
                delta = self._clip_delta_by_rho(last, delta)
                delta = self._apply_lda_guard(layer_id, last, delta)
                delta = self._apply_utility_guard(layer_id, last, delta)
                edited_last = last + delta
                edited = hidden.clone()
                edited[:, -1:, :] = edited_last
                return _replace_hidden(output, edited)

            self.handles.append(self.layers[layer_id].register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._decode_step_count = 0
        return False
