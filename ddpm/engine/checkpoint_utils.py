from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from ddpm.models.factory import build_unet
from ddpm.schedules.noise import DiffusionSchedule


def load_ddpm_model(ckpt_path: Path | str, device: torch.device | str):
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    metadata = ckpt.get("metadata")
    if metadata is None:
        raise ValueError("Checkpoint missing metadata required for reconstruction.")
    img_size = metadata.get("img_size", 32)
    arch = metadata.get("arch", "unet_small")
    model_params = metadata.get("model_params", {})
    model = build_unet(arch, img_size=img_size, overrides=model_params or {})
    state = ckpt.get("ema")
    if state is None:
        state = ckpt["model"]
    model.load_state_dict(state)
    schedule_cfg = metadata["diffusion"]
    schedule = DiffusionSchedule(T=schedule_cfg["T"], beta_schedule=schedule_cfg["beta_schedule"])
    return model.to(device), schedule, metadata

