from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "clone")


def apply_cmrm_to_hidden(output: Any, vector: Any, alpha: float, sign: int) -> Any:
    """Return output with only the last token hidden state shifted.

    Supports torch tensors in real generation and nested Python lists in tests.
    If a decoder layer returns a tuple, only tuple[0] is edited.
    """

    if isinstance(output, tuple):
        hidden = output[0]
        return (apply_cmrm_to_hidden(hidden, vector, alpha, sign),) + output[1:]

    if _is_torch_tensor(output):
        import torch

        hidden = output.clone()
        vec = vector if _is_torch_tensor(vector) else torch.tensor(vector)
        vec = vec.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] + sign * alpha * vec
        return hidden

    hidden = copy.deepcopy(output)
    vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    hidden[0][-1] = [x + sign * alpha * v for x, v in zip(hidden[0][-1], vec)]
    return hidden


class CMRMHookManager:
    def __init__(
        self,
        layers: list[Any],
        vectors: Mapping[int, Any],
        target_layers: list[int],
        alpha: float,
        sign: int,
        hook_scope: str = "all_steps",
    ) -> None:
        self.layers = layers
        self.vectors = vectors
        self.target_layers = target_layers
        self.alpha = alpha
        self.sign = sign
        self.hook_scope = hook_scope
        self.handles = []

    def __enter__(self):
        for layer_id in self.target_layers:
            vector = self.vectors[layer_id]

            def hook(_module, _inputs, output, layer_id=layer_id, vector=vector):
                hidden = output[0] if isinstance(output, tuple) else output
                shape = getattr(hidden, "shape", None)
                if shape is not None and len(shape) >= 2:
                    seq_len = shape[1]
                elif isinstance(hidden, list) and hidden and isinstance(hidden[0], list):
                    seq_len = len(hidden[0])
                else:
                    seq_len = None
                if self.hook_scope == "prefill_only" and seq_len == 1:
                    return output
                return apply_cmrm_to_hidden(output, vector, self.alpha, self.sign)

            self.handles.append(self.layers[layer_id].register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
