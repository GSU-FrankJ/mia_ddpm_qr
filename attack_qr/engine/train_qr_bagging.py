from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from attack_qr.models.qr_resnet18 import ResNet18QR
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
    def __init__(self, indexed_dataset: IndexedDataset, pairs_by_image: dict[int, dict[str, np.ndarray]]):
        self.dataset = indexed_dataset
        self.pairs_by_image = pairs_by_image
        self.image_ids = sorted(pairs_by_image.keys())
        self.idx_to_pos = {idx: pos for pos, idx in enumerate(self.dataset.indices)}
        if not self.image_ids:
            raise ValueError("No public samples available for quantile regression.")
        example = pairs_by_image[self.image_ids[0]][\"stats\"]
        self.stats_dim = example.shape[1]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, item: int):
        img_id = int(self.image_ids[item])
        pos = self.idx_to_pos[img_id]
        img, _, _ = self.dataset[pos]
        pair_info = self.pairs_by_image[img_id]
        choice = np.random.randint(0, pair_info[\"t_error\"].shape[0])
        stats = torch.tensor(pair_info[\"stats\"][choice], dtype=torch.float32)
        target = torch.tensor(pair_info[\"t_error\"][choice], dtype=torch.float32)
        return img, stats, target


def load_pairs(npz_path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    with np.load(npz_path) as data:
        image_ids = data[\"image_id\"].astype(np.int64)
        t_error = data[\"t_error\"].astype(np.float32)
        t_frac = data[\"t_frac\"].astype(np.float32)
        mean = data[\"mean\"].astype(np.float32)
        std = data[\"std\"].astype(np.float32)
        norm2 = data[\"norm2\"].astype(np.float32)

    pairs: dict[int, dict[str, np.ndarray]] = {}
    unique_ids = np.unique(image_ids)
    for img_id in unique_ids:
        mask = image_ids == img_id
        stats = np.stack([t_frac[mask], mean[mask], std[mask], norm2[mask]], axis=1)
        pairs[int(img_id)] = {
            \"t_error\": t_error[mask],
            \"stats\": stats,
        }
    return pairs


def prepare_dataset(
    dataset_name: str,
    root: str,
    public_indices: Sequence[int],
    img_size: int,
) -> IndexedDataset:
    base_dataset = get_dataset(dataset_name, root=root, download=True)
    transform = get_transforms(img_size, augment=False)
    return IndexedDataset(base_dataset, indices=public_indices, transform=transform)


def bootstrap_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=size, size=size)


def train_single_model(
    dataset: QuantilePairsDataset,
    config: QuantileTrainingConfig,
    model_seed: int,
    device: torch.device,
) -> tuple[ResNet18QR, List[float]]:
    seed_everything(model_seed)
    model = ResNet18QR(num_outputs=len(config.alpha_list), stats_dim=dataset.stats_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    alpha_tensor = torch.tensor(config.alpha_list, dtype=torch.float32, device=device)

    epoch_losses: List[float] = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for images, stats, targets in loader:
            images = images.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1).expand(-1, len(config.alpha_list))
            preds = model(images, stats)
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
    skip_existing: bool = False,
) -> None:
    seed_everything(config.seed)
    device = torch.device(device)
    pairs_by_image = load_pairs(npz_path)
    dataset = prepare_dataset(dataset_name, root=data_root, public_indices=public_indices, img_size=img_size)

    data = QuantilePairsDataset(dataset, pairs_by_image)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    base_manifest = {
        "dataset": dataset_name,
        "alpha_list": list(config.alpha_list),
        "bootstrap": config.bootstrap,
        "M": config.M,
        "seed": config.seed,
        "public_indices": [int(i) for i in public_indices],
        "stats_dim": data.stats_dim,
        "models": [],
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in base_manifest.items():
            if key == "models":
                continue
            if manifest.get(key) != value:
                raise ValueError(
                    f"Existing manifest at {manifest_path} has {key}={manifest.get(key)} but expected {value}."
                )
    else:
        manifest = base_manifest

    manifest.setdefault("models", [])
    manifest_lookup = {entry["path"]: entry for entry in manifest["models"]}

    def save_manifest() -> None:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if skip_existing and not manifest_path.exists():
        existing_ckpts = sorted(out_dir.glob("model_*.pt"))
        for ckpt_path in existing_ckpts:
            rel_path = ckpt_path.name
            if rel_path in manifest_lookup:
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            entry = {
                "path": rel_path,
                "seed": ckpt.get("seed"),
                "loss": (ckpt.get("losses") or [None])[-1],
            }
            if ckpt.get("bootstrap_indices") is not None:
                entry["bootstrap_indices"] = ckpt["bootstrap_indices"]
            if ckpt.get("bootstrap_image_ids") is not None:
                entry["bootstrap_image_ids"] = ckpt["bootstrap_image_ids"]
            manifest["models"].append(entry)
            manifest_lookup[rel_path] = entry
        if existing_ckpts:
            save_manifest()

    indices = np.arange(len(data))
    rng_master = np.random.default_rng(config.seed)

    for m in tqdm(range(config.M), desc="Bagging"):
        master_seed = int(rng_master.integers(0, 2**63 - 1))
        rng = np.random.default_rng(master_seed)
        ckpt_path = out_dir / f"model_{m:03d}.pt"
        rel_path = str(ckpt_path.relative_to(out_dir))
        existing_entry = manifest_lookup.get(rel_path)

        if skip_existing and ckpt_path.exists() and existing_entry is not None:
            continue
        if ckpt_path.exists() and not skip_existing:
            raise FileExistsError(f"Checkpoint {ckpt_path} already exists. Remove it or use --skip-existing.")

        if config.bootstrap:
            sampled_indices = bootstrap_indices(len(indices), rng)
            subset = Subset(data, sampled_indices.tolist())
            sampled_list = sampled_indices.tolist()
            sampled_image_ids = [int(data.image_ids[i]) for i in sampled_list]
        else:
            subset = data
            sampled_list = None
            sampled_image_ids = None

        model_seed = int(rng.integers(0, 2**31 - 1))
        seed_everything(model_seed)
        model, losses = train_single_model(subset, config, model_seed, device)
        torch.save(
            {
                "model": model.state_dict(),
                "alpha_list": list(config.alpha_list),
                "seed": model_seed,
                "losses": losses,
                "bootstrap_indices": sampled_list,
                "bootstrap_image_ids": sampled_image_ids,
            },
            ckpt_path,
        )
        entry = {
            "path": rel_path,
            "seed": model_seed,
            "loss": losses[-1] if losses else None,
        }
        if config.bootstrap:
            entry["bootstrap_indices"] = sampled_list
            entry["bootstrap_image_ids"] = sampled_image_ids

        manifest_lookup[rel_path] = entry
        manifest["models"] = [e for e in manifest["models"] if e["path"] != rel_path]
        manifest["models"].append(entry)
        save_manifest()
