from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = f[1:] / f[:-1]
    betas = 1 - alphas
    return torch.clamp(betas, min=1e-5, max=0.999)


def make_beta_schedule(name: Literal["linear", "cosine"], T: int) -> torch.Tensor:
    if name == "linear":
        return linear_beta_schedule(T)
    if name == "cosine":
        return cosine_beta_schedule(T)
    raise ValueError(f"Unknown beta schedule: {name}")


@dataclass
class DiffusionSchedule:
    T: int
    beta_schedule: str

    def __post_init__(self) -> None:
        betas = make_beta_schedule(self.beta_schedule, self.T)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffers(betas, alphas, alpha_cumprod)

    def register_buffers(self, betas: torch.Tensor, alphas: torch.Tensor, alpha_cumprod: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]], dim=0)
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas)
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.posterior_variance = betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod)

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        for name in [
            "betas",
            "alphas",
            "alpha_cumprod",
            "alpha_cumprod_prev",
            "sqrt_alphas",
            "sqrt_one_minus_alphas",
            "sqrt_alpha_cumprod",
            "sqrt_one_minus_alpha_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self


def extract(schedule_tensor: torch.Tensor, timesteps: torch.Tensor, x_shape) -> torch.Tensor:
    """
    Extract values from a precomputed tensor of size T using batch timesteps.
    """
    batch_size = timesteps.shape[0]
    out = schedule_tensor.gather(0, timesteps)
    return out.view(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(schedule: DiffusionSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    sqrt_alpha_cumprod = extract(schedule.sqrt_alpha_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_cumprod = extract(schedule.sqrt_one_minus_alpha_cumprod, t, x0.shape)
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise


def predict_x0(schedule: DiffusionSchedule, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
    sqrt_alpha_cumprod = extract(schedule.sqrt_alpha_cumprod, t, x_t.shape)
    sqrt_one_minus_alpha_cumprod = extract(schedule.sqrt_one_minus_alpha_cumprod, t, x_t.shape)
    return (x_t - sqrt_one_minus_alpha_cumprod * eps_pred) / sqrt_alpha_cumprod
