from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from attack_qr.models.qr_resnet_tiny import ResNetTiny
from attack_qr.utils.losses import pinball_loss
from attack_qr.utils.seeding import seed_everything
from ddpm.data.loader import IndexedDataset, get_dataset, get_transforms


@dataclass
class QuantileTrainingConfig:
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    alpha_list: Sequence[float] = (0.01, 0.001)
    bootstrap: bool = True
    M: int = 16
    seed: int = 0


class QuantilePairsDataset(Dataset):
    def __init__(self, indexed_dataset: IndexedDataset, image_ids: np.ndarray, targets: np.ndarray):
        self.dataset = indexed_dataset
        self.image_ids = image_ids.astype(np.int64)
        self.targets = targets.astype(np.float32)
        self.idx_to_pos = {idx: pos for pos, idx in enumerate(self.dataset.indices)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, item: int):
        idx = int(self.image_ids[item])
        target = float(self.targets[item])
        pos = self.idx_to_pos[idx]
        img, _, _ = self.dataset[pos]
        return img, torch.tensor(target, dtype=torch.float32)


def load_pairs(npz_path: str | Path) -> dict:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def prepare_dataset(
    dataset_name: str,
    root: str,
    public_indices: Sequence[int],
    img_size: int,
) -> IndexedDataset:
    base_dataset = get_dataset(dataset_name, root=root, download=True)
    transform = get_transforms(dataset_name, img_size, augment=False)
    return IndexedDataset(base_dataset, indices=public_indices, transform=transform)


def bootstrap_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=size, size=size)


def train_single_model(
    dataset: Dataset,
    config: QuantileTrainingConfig,
    model_seed: int,
    device: torch.device,
) -> tuple[ResNetTiny, List[float]]:
    seed_everything(model_seed)
    model = ResNetTiny(num_outputs=len(config.alpha_list)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    alpha_tensor = torch.tensor(config.alpha_list, dtype=torch.float32, device=device)

    epoch_losses: List[float] = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1).expand(-1, len(config.alpha_list))
            preds = model(images)
            loss = pinball_loss(preds, targets, alpha_tensor, reduction="mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = float(sum(losses) / max(1, len(losses)))
        epoch_losses.append(epoch_loss)
    return model, epoch_losses


def train_bagging_ensemble(
    npz_path: str | Path,
    dataset_name: str,
    public_indices: Sequence[int],
    config: QuantileTrainingConfig,
    out_dir: str | Path,
    img_size: int,
    data_root: str,
    device: str | torch.device = "cuda",
) -> None:
    seed_everything(config.seed)
    device = torch.device(device)
    pairs = load_pairs(npz_path)
    dataset = prepare_dataset(dataset_name, root=data_root, public_indices=public_indices, img_size=img_size)

    data = QuantilePairsDataset(dataset, pairs["image_id"], pairs["t_error"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset_name,
        "alpha_list": list(config.alpha_list),
        "bootstrap": config.bootstrap,
        "M": config.M,
        "seed": config.seed,
        "public_indices": [int(i) for i in public_indices],
        "models": [],
    }

    indices = np.arange(len(data))
    rng_master = np.random.default_rng(config.seed)

    for m in tqdm(range(config.M), desc="Bagging"):
        rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        if config.bootstrap:
            sampled_indices = bootstrap_indices(len(indices), rng)
            subset = Subset(data, sampled_indices.tolist())
        else:
            subset = data
        model_seed = int(rng.integers(0, 2**31 - 1))
        seed_everything(model_seed)
        model, losses = train_single_model(subset, config, model_seed, device)
        ckpt_path = out_dir / f"model_{m:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "alpha_list": list(config.alpha_list),
                "seed": model_seed,
                "losses": losses,
            },
            ckpt_path,
        )
        entry = {
            "path": str(ckpt_path.relative_to(out_dir)),
            "seed": model_seed,
            "loss": losses[-1] if losses else None,
        }
        if config.bootstrap:
            entry["bootstrap_indices"] = sampled_indices.tolist()
        manifest["models"].append(entry)

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
