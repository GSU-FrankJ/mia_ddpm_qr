"""DDIM training entrypoint for CIFAR-10."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def configure_environment(enable_amp: bool) -> Dict[str, object]:
    state: Dict[str, object] = {}

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        state["cudnn_allow_tf32"] = False
        if hasattr(torch.backends.cudnn, "fp32_math_mode") and hasattr(torch.backends.cudnn, "FP32MathMode"):
            try:
                torch.backends.cudnn.fp32_math_mode = torch.backends.cudnn.FP32MathMode.F32
            except AttributeError:
                pass
            state["cudnn_fp32_math_mode"] = "F32"
        else:
            state["cudnn_fp32_math_mode"] = None
        state["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        state["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    else:
        state["cudnn_benchmark"] = None
        state["cudnn_deterministic"] = None
        state["cudnn_allow_tf32"] = None
        state["cudnn_fp32_math_mode"] = None

    if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul_backend = torch.backends.cuda.matmul
        if hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False
            state["cuda_matmul_allow_tf32"] = False
        else:
            state["cuda_matmul_allow_tf32"] = None
        if hasattr(matmul_backend, "fp32_precision"):
            try:
                matmul_backend.fp32_precision = "ieee"
            except AttributeError:
                pass
            state["cuda_matmul_fp32_precision"] = "ieee"
        else:
            state["cuda_matmul_fp32_precision"] = None
    else:
        state["cuda_matmul_allow_tf32"] = None
        state["cuda_matmul_fp32_precision"] = None

    torch.set_float32_matmul_precision("high")
    state["float32_matmul_precision"] = torch.get_float32_matmul_precision()

    amp_enabled = enable_amp and torch.cuda.is_available()
    state["amp_enabled"] = amp_enabled
    state["amp_mode"] = "torch.amp.autocast" if amp_enabled else "disabled"
    state["amp_device"] = "cuda" if amp_enabled else "cpu"
    return state


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def diffusion_training_step(
    model: UNetModel,
    batch: torch.Tensor,
    alphas_bar: torch.Tensor,
    T: int,
    scaler: torch.amp.GradScaler | None,
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

    device_type = "cuda" if batch.is_cuda else "cpu"
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, enabled=scaler is not None and scaler.is_enabled())
        if device_type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        pred_noise = model(xt, timesteps)
        loss = F.mse_loss(pred_noise, noise)

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
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
    determinism_state: Dict[str, object],
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
            "determinism": determinism_state,
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
    determinism_state = configure_environment(model_cfg["training"].get("amp", True))

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

    use_cuda_amp = torch.cuda.is_available() and model_cfg["training"].get("amp", True)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if use_cuda_amp
        else None
    )

    T = model_cfg["diffusion"]["timesteps"]
    _betas, alphas_bar = build_cosine_schedule(T)
    alphas_bar = alphas_bar.to(device)

    iterations = model_cfg["training"]["iterations"][args.mode]
    if args.mode == "fastdev":
        fastdev_limit = model_cfg["training"].get("fastdev_limit")
        if fastdev_limit is not None:
            iterations = min(iterations, fastdev_limit)
    checkpoint_interval = model_cfg["training"]["checkpoint_interval"]

    run_dir = pathlib.Path(model_cfg["experiment"]["output_dir"]) / args.mode
    write_run_metadata(run_dir, model_cfg, data_cfg, seed, args.mode, determinism_state)

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

