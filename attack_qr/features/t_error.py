from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from attack_qr.utils.seeding import make_generator, philox_seed
from ddpm.schedules.noise import DiffusionSchedule, predict_x0, q_sample


def _batched_noise(shape: torch.Size, seeds: Sequence[int], device: torch.device) -> torch.Tensor:
    noises = []
    for seed in seeds:
        gen = make_generator(seed, device=device)
        noises.append(torch.randn(shape[1:], device=device, generator=gen))
    return torch.stack(noises, dim=0)


def compute_t_error(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    dataset_name: str,
    sample_indices: Sequence[int],
    global_seed: int,
    mode: str = "eps",
) -> torch.Tensor:
    """
    Compute deterministic t-error for a batch of images.
    """
    assert mode in {"eps", "x0"}
    device = x0.device
    model.eval()

    seeds = [philox_seed(dataset_name, idx, int(t.item()), global_seed) for idx, t in zip(sample_indices, timesteps)]
    noise = _batched_noise(x0.shape, seeds, device=device)
    with torch.no_grad():
        x_t = q_sample(schedule, x0, timesteps, noise)
        eps_pred = model(x_t, timesteps)
        if mode == "eps":
            loss = F.mse_loss(eps_pred, noise, reduction="none")
        else:
            x0_pred = predict_x0(schedule, x_t, timesteps, eps_pred)
            loss = F.mse_loss(x0_pred, x0, reduction="none")
    dims = list(range(1, loss.ndim))
    return loss.mean(dim=dims)
