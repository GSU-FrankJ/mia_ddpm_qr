from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack_qr.features.t_error import compute_t_error
from attack_qr.utils.seeding import timesteps_seed
from ddpm.schedules.noise import DiffusionSchedule


def build_t_error_pairs(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    dataset_name: str,
    global_seed: int,
    K: int,
    mode: str,
    out_path: str | Path,
    device: str | torch.device = "cuda",
) -> Dict[str, int]:
    model = model.to(device)
    schedule = schedule.to(device)
    model.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = {
        "image_id": [],
        "timestep": [],
        "t_frac": [],
        "t_error": [],
        "mean": [],
        "std": [],
        "norm2": [],
    }
    total_pairs = 0

    for images, _, indices in tqdm(dataloader, desc="Pairs"):
        images = images.to(device)
        for img, idx in zip(images, indices):
            idx_int = int(idx)
            stats_mean = float(img.mean().item())
            stats_std = float(img.std(unbiased=False).item())
            stats_norm2 = float(torch.linalg.norm(img.float()).item())
            rng_seed = timesteps_seed(dataset_name, idx_int, global_seed)
            rng = np.random.default_rng(rng_seed)
            timesteps = rng.integers(low=0, high=schedule.T, size=K, endpoint=False)
            x_batch = img.unsqueeze(0).repeat(K, 1, 1, 1)
            t_tensor = torch.as_tensor(timesteps, device=device, dtype=torch.long)
            sample_indices = [idx_int] * K
            errors = compute_t_error(
                model=model,
                schedule=schedule,
                x0=x_batch,
                timesteps=t_tensor,
                dataset_name=dataset_name,
                sample_indices=sample_indices,
                global_seed=global_seed,
                mode=mode,
            )
            for t_val, err in zip(timesteps.tolist(), errors.detach().cpu().tolist()):
                records["image_id"].append(idx_int)
                records["timestep"].append(int(t_val))
                records["t_frac"].append(float(t_val / max(1, schedule.T - 1)))
                records["t_error"].append(float(err))
                records["mean"].append(stats_mean)
                records["std"].append(stats_std)
                records["norm2"].append(stats_norm2)
                total_pairs += 1

    np.savez_compressed(
        out_path,
        image_id=np.array(records["image_id"], dtype=np.int32),
        timestep=np.array(records["timestep"], dtype=np.int32),
        t_frac=np.array(records["t_frac"], dtype=np.float32),
        t_error=np.array(records["t_error"], dtype=np.float32),
        mean=np.array(records["mean"], dtype=np.float32),
        std=np.array(records["std"], dtype=np.float32),
        norm2=np.array(records["norm2"], dtype=np.float32),
    )

    metadata = {
        "dataset": dataset_name,
        "global_seed": global_seed,
        "mode": mode,
        "K": K,
        "T": schedule.T,
        "pairs": total_pairs,
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata
