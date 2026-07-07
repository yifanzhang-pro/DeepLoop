from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn

_DEFAULT_EXCLUDE_SUFFIXES = (
    # Positional-encoding params that have their own init (e.g., GRAPE freq spectra).
    "grape.log_freq",
)
_DEFAULT_EXCLUDE_NAMES = (
    "transformer.wte.weight",
    "lm_head.weight",
)
_RESIDUAL_OUTPUT_PROJ_SUFFIXES = (
    "attn.o_proj.weight",
    "mlp.o_proj.weight",
)


def _should_exclude_parameter(name: str, *, exclude_name_set: set[str], exclude_suffixes: tuple[str, ...]) -> bool:
    if name in exclude_name_set:
        return True
    return any(name.endswith(suffix) for suffix in exclude_suffixes)


def _spectral_std(*, n_in: int, n_out: int, hidden_init_std_factor: float) -> float:
    if n_in <= 0:
        raise ValueError(f"n_in must be positive, got {n_in}.")
    if n_out <= 0:
        raise ValueError(f"n_out must be positive, got {n_out}.")
    if hidden_init_std_factor < 0.0:
        raise ValueError(f"hidden_init_std_factor must be >= 0, got {hidden_init_std_factor}.")
    return (hidden_init_std_factor / math.sqrt(float(n_in))) * min(1.0, math.sqrt(float(n_out) / float(n_in)))


def _tensor_fans(param: torch.Tensor) -> tuple[int, int]:
    if param.dim() < 2:
        raise ValueError(f"Expected tensor with rank >= 2, got rank {param.dim()}.")
    # Treat the first axis as n_out and all remaining axes as n_in.
    # This matches linear weights (out, in) and Conv1d weights
    # (out_channels, in_channels/groups, kernel_size).
    n_out = int(param.shape[0])
    if n_out <= 0:
        raise ValueError(f"Expected first dimension to be positive, got {n_out}.")
    n_in = int(param.numel()) // n_out
    if n_in <= 0:
        raise ValueError(f"Computed n_in must be positive, got {n_in}.")
    return n_in, n_out


def _is_residual_output_projection(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _RESIDUAL_OUTPUT_PROJ_SUFFIXES)


def init_gpt_weights(
    model: nn.Module,
    config,
    *,
    exclude_suffixes: tuple[str, ...] = (),
    exclude_names: tuple[str, ...] = (),
) -> None:
    init_std = float(getattr(config, "embedding_init_std", 0.02))
    hidden_init_std_factor = float(getattr(config, "hidden_init_std_factor", 0.5))
    num_layers = int(getattr(config, "num_hidden_layers"))
    if num_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_layers}.")

    exclude_suffixes = _DEFAULT_EXCLUDE_SUFFIXES + tuple(exclude_suffixes)
    exclude_name_set = set(_DEFAULT_EXCLUDE_NAMES + tuple(exclude_names))

    with torch.no_grad():
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            lm_head.weight.normal_(mean=0.0, std=init_std)

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if _should_exclude_parameter(name, exclude_name_set=exclude_name_set, exclude_suffixes=exclude_suffixes):
                continue
            n_in, n_out = _tensor_fans(param)
            std = _spectral_std(
                n_in=n_in,
                n_out=n_out,
                hidden_init_std_factor=hidden_init_std_factor,
            )
            if _is_residual_output_projection(name):
                std = std / float(num_layers)
            param.normal_(mean=0.0, std=std)


def init_test_randn(
    base_randn: Callable[..., torch.Tensor],
    *size: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Route test normal initialization through `model.init_utils`.

    This indirection lets tests use a centralized initialization entrypoint
    without changing the distribution semantics of `torch.randn`.
    """

    return base_randn(*size, **kwargs)


def init_test_rand(
    base_rand: Callable[..., torch.Tensor],
    *size: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Route test uniform initialization through `model.init_utils`.

    This indirection lets tests use a centralized initialization entrypoint
    without changing the distribution semantics of `torch.rand`.
    """

    return base_rand(*size, **kwargs)
