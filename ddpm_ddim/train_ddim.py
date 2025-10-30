"""DDIM training entrypoint for CIFAR-10."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import time
from dataclasses import asdict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import yaml

from mia_logging import get_winston_logger
from ddpm_ddim.models.unet import UNetModel, UNetConfig, build_unet
from ddpm_ddim.schedulers.betas import build_cosine_schedule


LOGGER = get_winston_logger(__name__)


class CIFAR10Subset(Dataset):
    """CIFAR-10 subset wrapper with deterministic transforms."""

    def __init__(
        self,
        root: pathlib.Path,
        indices: torch.Tensor,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dataset = CIFAR10(root=str(root), train=True, download=False, transform=transform)
        self.subset = Subset(dataset, indices.tolist())

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        return self.subset[idx]


class EMA:
    """Exponential moving average helper for model weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.ema_model = build_unet(asdict(model.config))  # type: ignore[arg-type]
        self.ema_model.load_state_dict(model.state_dict())
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def to(self, device: torch.device) -> None:
        self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def diffusion_training_step(
    model: UNetModel,
    batch: torch.Tensor,
    alphas_bar: torch.Tensor,
    T: int,
    scaler: amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    device = batch.device
    bsz = batch.size(0)
    timesteps = torch.randint(0, T, (bsz,), device=device, dtype=torch.long)
    noise = torch.randn_like(batch)
    alpha_bar_t = alphas_bar[timesteps]
    sqrt_alpha = alpha_bar_t.sqrt()[:, None, None, None]
    sqrt_one_minus = (1 - alpha_bar_t).sqrt()[:, None, None, None]
    xt = sqrt_alpha * batch + sqrt_one_minus * noise

    with amp.autocast(enabled=scaler.is_enabled()):
        pred_noise = model(xt, timesteps)
        loss = F.mse_loss(pred_noise, noise)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def save_checkpoint(
    model: UNetModel,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_dir: pathlib.Path,
) -> None:
    ckpt_dir = run_dir / f"ckpt_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "state_dict": model.state_dict()}, ckpt_dir / "model.ckpt")
    torch.save({"step": step, "state_dict": ema.ema_model.state_dict()}, ckpt_dir / "ema.ckpt")
    torch.save({"step": step, "state_dict": optimizer.state_dict()}, ckpt_dir / "optim.ckpt")
    LOGGER.info("Checkpoint saved at step %d -> %s", step, ckpt_dir)


def write_run_metadata(
    run_dir: pathlib.Path,
    model_cfg: Dict,
    data_cfg: Dict,
    seed: int,
    mode: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run_dir)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        LOGGER.warning("Unable to resolve git commit hash")

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": git_hash,
        "seed": seed,
        "mode": mode,
        "model_config": model_cfg,
        "data_config": data_cfg,
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_arch": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with (run_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    LOGGER.info("Wrote run metadata to %s", run_dir / "run.json")


def load_indices(path: pathlib.Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return torch.tensor(data, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDIM on CIFAR-10")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/model_ddim.yaml"))
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--mode", choices=["main", "fastdev"], default="main")
    parser.add_argument("--fastdev", action="store_true", help="Alias for --mode fastdev")
    args = parser.parse_args()

    if args.fastdev:
        args.mode = "fastdev"

    model_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)

    seed = model_cfg.get("seed", 0)
    set_global_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"].get("member_train", "data/splits/member_train.json")))
    if args.mode == "fastdev":
        train_indices = train_indices[:1024]

    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])  # type: ignore[assignment]
    std = tuple(data_cfg["dataset"]["normalization"]["std"])  # type: ignore[assignment]

    dataset = CIFAR10Subset(root=root, indices=train_indices, mean=mean, std=std)
    batch_size = model_cfg["training"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg["dataset"].get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )

    model = build_unet(model_cfg["model"])
    model.to(device)
    ema = EMA(model, decay=model_cfg["training"]["ema_decay"])
    ema.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["training"]["lr"],
        betas=tuple(model_cfg["training"]["betas"]),
        weight_decay=model_cfg["training"].get("weight_decay", 0.0),
    )

    scaler = amp.GradScaler(enabled=model_cfg["training"].get("amp", True))

    T = model_cfg["diffusion"]["timesteps"]
    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    iterations = model_cfg["training"]["iterations"][args.mode]
    checkpoint_interval = model_cfg["training"]["checkpoint_interval"]

    run_dir = pathlib.Path(model_cfg["experiment"]["output_dir"]) / args.mode
    write_run_metadata(run_dir, model_cfg, data_cfg, seed, args.mode)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < iterations:
        for batch, _ in dataloader:
            batch = batch.to(device)
            loss = diffusion_training_step(
                model,
                batch,
                alphas_bar,
                T,
                scaler,
                optimizer,
                model_cfg["training"]["grad_clip"],
            )
            ema.update(model)
            step += 1
            if step % model_cfg["training"]["log_interval"] == 0:
                LOGGER.info("step=%d loss=%.4f", step, loss)
            if step % checkpoint_interval == 0:
                save_checkpoint(model, ema, optimizer, step, run_dir)
            if step >= iterations:
                break

    save_checkpoint(model, ema, optimizer, step, run_dir)
    LOGGER.info("Training finished at step %d", step)


if __name__ == "__main__":
    main()

