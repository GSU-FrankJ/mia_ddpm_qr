"""Noise schedule utilities for DDPM/DDIM."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine cumulative product schedule introduced by Nichol & Dhariwal.

    Args:
        t: Normalised timesteps in `[0, 1]` as a float tensor.
        s: Small offset that avoids division by zero at the endpoints.

    Returns:
        Alpha-bar values matching the integral of the cosine schedule.
    """

    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


def build_cosine_schedule(T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct beta and alpha-bar tensors for a cosine diffusion schedule.

    Args:
        T: Number of diffusion steps.

    Returns:
        betas: Noise variance increments shaped `[T]`.
        alphas_bar: Cumulative product of `(1 - beta)` shaped `[T]`.
    """

    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_bar = cosine_alpha_bar(steps / T)
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(min=1e-5, max=0.999)
    LOGGER.debug("Constructed cosine schedule with T=%s", T)
    return betas.float(), alphas_bar[1:].float()


__all__ = ["build_cosine_schedule"]


