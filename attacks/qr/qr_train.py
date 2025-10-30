"""Training utilities for quantile regression models."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mia_logging import get_winston_logger
from attacks.qr.qr_models import SmallCNNQuantile, pinball_loss


LOGGER = get_winston_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    tau: float
    device: torch.device
    early_stop_patience: int = 10
    grad_clip: float = 1.0
    use_log1p: bool = True
    warmup_tau: float | None = None
    warmup_epochs: int = 0


@dataclass
class TrainHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


def train_quantile_model(
    model: SmallCNNQuantile,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> Dict:
    model.to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = 0
    history = TrainHistory()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        tau_active = cfg.tau
        if cfg.warmup_tau is not None and epoch < cfg.warmup_epochs:
            tau_active = cfg.warmup_tau

        for batch in train_loader:
            if len(batch) == 2:
                images, target_raw = batch
                target_log = torch.log1p(target_raw.clamp_min(0))
            else:
                images, target_raw, target_log = batch
            images = images.to(cfg.device)
            targets = target_log.to(cfg.device) if cfg.use_log1p else target_raw.to(cfg.device)
            preds = model(images)
            loss = pinball_loss(preds, targets, tau_active)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        history.train_losses.append(epoch_loss)

        model.eval()
        val_dataset_size = len(val_loader.dataset)
        if val_dataset_size > 0:
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 2:
                        images, target_raw = batch
                        target_log = torch.log1p(target_raw.clamp_min(0))
                    else:
                        images, target_raw, target_log = batch
                    images = images.to(cfg.device)
                    targets = target_log.to(cfg.device) if cfg.use_log1p else target_raw.to(cfg.device)
                    preds = model(images)
                    val_loss += pinball_loss(preds, targets, tau_active).item() * images.size(0)
            val_loss /= val_dataset_size
        else:
            val_loss = epoch_loss
        history.val_losses.append(val_loss)

        LOGGER.info(
            "epoch=%d tau=%.5f train_loss=%.6f val_loss=%.6f",
            epoch,
            tau_active,
            epoch_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                LOGGER.info("Early stopping triggered for tau=%.5f", tau_active)
                break

    return {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "history": history,
        "best_val": best_val,
    }


