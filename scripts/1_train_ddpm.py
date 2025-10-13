#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ddpm.data.loader import dataloader_from_indices, get_dataset, load_split_indices
from ddpm.engine.train_ddpm import OptimConfig, TrainConfig, train_ddpm
from ddpm.models.factory import build_unet
from ddpm.schedules.noise import DiffusionSchedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDPM model for QR-MIA pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to DDPM YAML config.")
    parser.add_argument("--split_json", type=str, required=True, help="Split JSON from data preparation.")
    parser.add_argument("--out", type=str, default=None, help="Override output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda or cpu).")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dataset_name = config.get("dataset")
    if dataset_name is None:
        raise ValueError("Config must specify 'dataset'.")
    img_size = config.get("img_size", 32)
    data_root = config.get("data_root", config.get("paths", {}).get("data_root", "data"))
    arch_config = config

    train_cfg = config["train"]
    diffusion_cfg = config["diffusion"]

    split_info = load_split_indices(args.split_json)
    z_indices = split_info["z"]

    train_loader = dataloader_from_indices(
        dataset_name=dataset_name,
        base_dataset=get_dataset(dataset_name, root=data_root, download=True),
        indices=z_indices,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 4),
        shuffle=True,
        seed=train_cfg.get("seed", 0),
        img_size=img_size,
        augment=True,
    )

    model = build_unet(arch_config.get("arch", "unet_small"), img_size=img_size, overrides=arch_config.get("model"))
    schedule = DiffusionSchedule(T=diffusion_cfg["T"], beta_schedule=diffusion_cfg["beta_schedule"])

    optim_config = OptimConfig(
        lr=train_cfg["lr"],
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    train_config = TrainConfig(
        epochs=train_cfg["epochs"],
        grad_clip=train_cfg.get("grad_clip", 1.0),
        ema=train_cfg.get("ema", True),
        ema_decay=train_cfg.get("ema_decay", 0.999),
        seed=train_cfg.get("seed", 0),
    )

    default_out = config.get("logging", {}).get("out_dir", f"runs/ddpm/{dataset_name}")
    out_dir = Path(args.out) if args.out else Path(default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(Path(args.config).resolve()),
        "split_json": str(Path(args.split_json).resolve()),
        "dataset": dataset_name,
        "img_size": img_size,
        "diffusion": diffusion_cfg,
        "arch": arch_config.get("arch", "unet_small"),
        "model_params": arch_config.get("model", {}),
    }

    summary = train_ddpm(
        model=model,
        schedule=schedule,
        dataloader=train_loader,
        optim_config=optim_config,
        train_config=train_config,
        out_dir=out_dir,
        device=torch.device(args.device),
        log_interval=config.get("logging", {}).get("log_interval", 100),
        metadata=metadata,
    )

    (out_dir / "ddpm_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata | summary, f, indent=2)


if __name__ == "__main__":
    main()
