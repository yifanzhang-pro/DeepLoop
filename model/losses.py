"""Loss functions, including HRM's stablemax cross-entropy.

stablemax avoids the standard softmax exp() and uses a numerically stable
positive transform: s(x) = 1/(1-x+eps) for x<0, else x+1. The loss matches
the cross-entropy structure but in float64, which HRM finds beneficial for
ARC-AGI training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = _s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Mean stablemax cross-entropy over non-ignored labels (HRM style).

    `logits`: (..., V), `labels`: (...) with values in [0, V) or `ignore_index`.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    logprobs = log_stablemax(flat_logits.to(torch.float64), dim=-1)
    valid_mask = flat_labels != ignore_index
    safe_labels = torch.where(valid_mask, flat_labels, torch.zeros_like(flat_labels))
    pred_logp = torch.gather(logprobs, dim=-1, index=safe_labels.long().unsqueeze(-1)).squeeze(-1)
    losses = -torch.where(valid_mask, pred_logp, torch.zeros_like(pred_logp))
    n_valid = valid_mask.sum().clamp_min(1)
    return losses.sum() / n_valid


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Standard softmax cross-entropy, mean reduction over non-ignored labels."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).to(torch.float32),
        labels.reshape(-1).long(),
        ignore_index=ignore_index,
    )


def get_loss_fn(loss_type: str):
    if loss_type == "stablemax_ce":
        return stablemax_cross_entropy
    if loss_type == "softmax_ce":
        return softmax_cross_entropy
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
