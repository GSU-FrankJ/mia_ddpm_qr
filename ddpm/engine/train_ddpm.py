from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack_qr.utils.seeding import seed_everything
from ddpm.schedules.noise import DiffusionSchedule, q_sample


@dataclass
class OptimConfig:
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 200
    grad_clip: Optional[float] = 1.0
    ema: bool = True
    ema_decay: float = 0.999
    seed: int = 0


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float):
        import copy

        self.model = copy.deepcopy(model)
        self.decay = decay
        self.model.requires_grad_(False)

    def to(self, device: torch.device | str) -> "EMAModel":
        self.model.to(device)
        return self

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            ema_params = dict(self.model.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                ema_param = ema_params[name]
                ema_param.mul_(self.decay).add_(param, alpha=1 - self.decay)
            for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)


def create_optimizer(model: torch.nn.Module, config: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def train_ddpm(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    optim_config: OptimConfig,
    train_config: TrainConfig,
    out_dir: str | Path,
    device: torch.device | str = "cuda",
    log_interval: int = 100,
    metadata: Optional[dict] = None,
) -> Dict[str, float]:
    seed_everything(train_config.seed)
    device = torch.device(device)
    model.to(device)
    schedule = schedule.to(device)

    optimizer = create_optimizer(model, optim_config)
    ema_helper = EMAModel(model, decay=train_config.ema_decay).to(device) if train_config.ema else None

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_csv = out_path / "train_log.csv"

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss"])

    global_step = 0
    best_loss = float("inf")
    best_state: Optional[dict] = None

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        epoch_losses = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in progress:
            x, _, _ = batch  # (image, label, index)
            x = x.to(device)
            noise = torch.randn_like(x)
            bsz = x.size(0)
            t = torch.randint(0, schedule.T, (bsz,), device=device, dtype=torch.long)
            x_t = q_sample(schedule, x, t, noise)

            preds = model(x_t, t)
            loss = F.mse_loss(preds, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            if ema_helper is not None:
                ema_helper.update(model)

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1
            if global_step % log_interval == 0:
                progress.set_postfix({"loss": f"{loss_val:.4f}"})

        epoch_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        with log_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, global_step, epoch_loss])

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                "model": model.state_dict(),
                "ema": ema_helper.model.state_dict() if ema_helper is not None else None,
                "epoch": epoch,
                "step": global_step,
                "loss": epoch_loss,
                "train_config": train_config.__dict__,
                "optim_config": optim_config.__dict__,
            }

        checkpoint = {
            "model": model.state_dict(),
            "ema": ema_helper.model.state_dict() if ema_helper is not None else None,
            "epoch": epoch,
            "step": global_step,
            "train_config": train_config.__dict__,
            "optim_config": optim_config.__dict__,
            "metadata": metadata,
        }
        torch.save(checkpoint, out_path / f"epoch_{epoch:04d}.pt")

    if best_state is not None:
        if metadata is not None:
            best_state["metadata"] = metadata
        torch.save(best_state, out_path / "best.pt")

    summary = {
        "best_loss": best_loss,
        "best_epoch": best_state["epoch"] if best_state is not None else train_config.epochs,
        "final_epoch": train_config.epochs,
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
