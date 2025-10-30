"""Deterministic DDIM forward and reverse mappings."""

from __future__ import annotations

from typing import Callable

import torch

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


def _extract(alphas_bar: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Gather ᾱₜ values for each sample in the batch."""

    return alphas_bar.to(timesteps.device)[timesteps]


def ddim_forward(
    x0: torch.Tensor,
    t: torch.Tensor,
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alphas_bar: torch.Tensor,
) -> torch.Tensor:
    """Apply the deterministic DDIM forward mapping.

    Args:
        x0: Clean images `[B, 3, 32, 32]`.
        t: Integer timesteps `[B]` with values `< T`.
        model: Diffusion model (unused but included for signature parity).
        alphas_bar: ᾱ cumulative schedule `[T]`.

    Returns:
        xt: Noised latent at timestep `t` produced without stochasticity.
    """

    alpha_bar_t = _extract(alphas_bar, t)
    sqrt_alpha = alpha_bar_t.sqrt()[:, None, None, None]
    sqrt_one_minus = (1 - alpha_bar_t).sqrt()[:, None, None, None]
    # Deterministic DDIM forward uses η=0, i.e. the mean of q(x_t | x_0).
    # This keeps membership scores reproducible across runs and removes
    # stochastic noise that would otherwise dominate the t-error signal.
    xt = sqrt_alpha * x0
    if torch.any(torch.isnan(xt)):
        LOGGER.error("Encountered NaNs in DDIM forward pass")
        raise FloatingPointError("NaNs in DDIM forward computation")
    return xt


def ddim_reverse(
    xt: torch.Tensor,
    t: torch.Tensor,
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alphas_bar: torch.Tensor,
) -> torch.Tensor:
    """Apply the deterministic DDIM reverse mapping.

    Args:
        xt: Latents at timestep `t` shaped `[B, 3, 32, 32]`.
        t: Integer timesteps `[B]`.
        model: Denoiser predicting εₜ given `(xₜ, t)`.
        alphas_bar: ᾱ schedule `[T]`.

    Returns:
        x0_hat: Reconstructed images `[B, 3, 32, 32]`.
    """

    alpha_bar_t = _extract(alphas_bar, t)
    sqrt_alpha = alpha_bar_t.sqrt()[:, None, None, None]
    sqrt_one_minus = (1 - alpha_bar_t).sqrt()[:, None, None, None]
    eps_pred = model(xt, t)
    x0_hat = (xt - sqrt_one_minus * eps_pred) / sqrt_alpha
    return x0_hat


__all__ = ["ddim_forward", "ddim_reverse"]

