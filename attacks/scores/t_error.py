"""t-error score implementation."""

from __future__ import annotations

from typing import Callable, List, Sequence

import torch

from mia_logging import get_winston_logger
from ddpm_ddim.ddim import ddim_forward, ddim_reverse


LOGGER = get_winston_logger(__name__)


def uniform_timesteps(T: int = 1000, k: int = 50) -> List[int]:
    """Return `k` approximately uniform timestep indices on `[0, T)`."""

    if k <= 0:
        raise ValueError("k must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    grid = torch.linspace(0, T - 1, steps=k)
    return grid.round().long().tolist()


def t_error_once(
    x0: torch.Tensor,
    t: int,
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alphas_bar: torch.Tensor,
) -> torch.Tensor:
    """Compute deterministic forward→reverse reconstruction error at step `t`.

    Args:
        x0: Clean images `[B, 3, 32, 32]`.
        t: Scalar timestep.
        model: Denoising network.
        alphas_bar: ᾱ schedule `[T]`.

    Returns:
        Tensor `[B]` with squared L2 errors.
    """

    if x0.dim() != 4:
        raise ValueError("x0 must be [B, C, H, W]")

    device = x0.device
    timesteps = torch.full((x0.size(0),), t, device=device, dtype=torch.long)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=x0.is_cuda):
        xt = ddim_forward(x0, timesteps, model, alphas_bar)
        x_hat = ddim_reverse(xt, timesteps, model, alphas_bar)
    error = (x_hat - x0).pow(2).flatten(start_dim=1).sum(dim=1)
    LOGGER.debug("t-error step=%d mean=%.6f", t, error.mean().item())
    return error


def t_error_aggregate(
    x0: torch.Tensor,
    T_set: Sequence[int],
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alphas_bar: torch.Tensor,
    agg: str = "mean",
) -> torch.Tensor:
    """Aggregate t-errors across a set of timesteps.

    Args:
        x0: Clean images `[B, 3, 32, 32]`.
        T_set: Iterable of timestep indices.
        model: Denoiser.
        alphas_bar: ᾱ schedule `[T]`.
        agg: Aggregation strategy (`mean`, `min`, `q10`).
    """

    errors = []
    for t in T_set:
        errors.append(t_error_once(x0, t, model, alphas_bar))
    stack = torch.stack(errors, dim=1)
    if agg == "mean":
        return stack.mean(dim=1)
    if agg == "min":
        return stack.min(dim=1).values
    if agg.startswith("q"):
        q = float(agg[1:]) / 100 if agg != "q10" else 0.10
        return torch.quantile(stack, q=q, dim=1)
    raise ValueError(f"Unsupported aggregation: {agg}")


__all__ = ["uniform_timesteps", "t_error_once", "t_error_aggregate"]

