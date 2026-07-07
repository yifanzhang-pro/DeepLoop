from __future__ import annotations

from collections.abc import Mapping

import torch

_C_QKV_WEIGHT_SUFFIX = ".attn.c_qkv.weight"
_C_QKV_BIAS_SUFFIX = ".attn.c_qkv.bias"


def _split_qkv_tensor(tensor: torch.Tensor, *, key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor.shape[0] % 3 != 0:
        raise ValueError(
            f"Legacy fused QKV tensor {key!r} has shape {tuple(tensor.shape)}; expected dim0 to be divisible by 3."
        )
    q, k, v = tensor.chunk(3, dim=0)
    return q, k, v


def remap_legacy_c_qkv_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    """
    Convert legacy fused `c_qkv` keys to split `q_proj`/`k_proj`/`v_proj` keys.

    Returns a tuple: `(remapped_state_dict, did_remap)`.
    """
    remapped = False
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_C_QKV_WEIGHT_SUFFIX):
            prefix = key.removesuffix("c_qkv.weight")
            q, k, v = _split_qkv_tensor(tensor, key=key)
            out[f"{prefix}q_proj.weight"] = q
            out[f"{prefix}k_proj.weight"] = k
            out[f"{prefix}v_proj.weight"] = v
            remapped = True
            continue
        if key.endswith(_C_QKV_BIAS_SUFFIX):
            prefix = key.removesuffix("c_qkv.bias")
            q, k, v = _split_qkv_tensor(tensor, key=key)
            out[f"{prefix}q_proj.bias"] = q
            out[f"{prefix}k_proj.bias"] = k
            out[f"{prefix}v_proj.bias"] = v
            remapped = True
            continue
        out[key] = tensor
    return out, remapped
