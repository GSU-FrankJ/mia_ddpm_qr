from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ddpm.models.unet import UNetModel
from ddpm.schedules.noise import DiffusionSchedule, extract, predict_x0


@dataclass
class DDPMCheckpoint:
    model_state: dict
    ema_state: Optional[dict]
    epoch: int
    step: int


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> DDPMCheckpoint:
    ckpt = torch.load(path, map_location=map_location)
    return DDPMCheckpoint(
        model_state=ckpt["model"],
        ema_state=ckpt.get("ema"),
        epoch=ckpt.get("epoch", -1),
        step=ckpt.get("step", -1),
    )


def restore_model(
    model: UNetModel,
    checkpoint: DDPMCheckpoint,
    use_ema: bool = True,
    device: str | torch.device = "cpu",
) -> UNetModel:
    model.load_state_dict(checkpoint.ema_state if use_ema and checkpoint.ema_state is not None else checkpoint.model_state)
    return model.to(device)


def p_mean_variance(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = model(x, t)
    x0_pred = predict_x0(schedule, x, t, eps)
    coef1 = extract(schedule.posterior_mean_coef1, t, x.shape)
    coef2 = extract(schedule.posterior_mean_coef2, t, x.shape)
    mean = coef1 * x0_pred + coef2 * x
    var = extract(schedule.posterior_variance, t, x.shape)
    log_var = extract(schedule.posterior_log_variance_clipped, t, x.shape)
    return mean, var, log_var, eps, x0_pred


def p_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x: torch.Tensor,
    t: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    mean, var, log_var, _, _ = p_mean_variance(model, schedule, x, t)
    noise = torch.randn_like(x, generator=generator)
    nonzero_mask = (t != 0).float().view(-1, *((1,) * (x.ndim - 1)))
    return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise


def sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    device: str | torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=device, generator=generator)
        for t_idx in reversed(range(schedule.T)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x = p_sample(model, schedule, x, t, generator=generator)
        return x

