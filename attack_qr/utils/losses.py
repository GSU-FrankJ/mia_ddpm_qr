from __future__ import annotations

from typing import Iterable, Literal, Sequence

import torch


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float | Sequence[float] | torch.Tensor,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute the pinball loss for quantile regression.
    """
    alpha_tensor = torch.as_tensor(alpha, device=pred.device, dtype=pred.dtype)
    while alpha_tensor.ndim < pred.ndim:
        alpha_tensor = alpha_tensor.unsqueeze(0)
    diff = target - pred
    loss = torch.maximum(alpha_tensor * diff, (alpha_tensor - 1.0) * diff)
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")

