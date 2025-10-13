from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from attack_qr.utils.seeding import make_generator


DATASET_STATS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
}


def get_dataset(dataset_name: str, root: str, download: bool = True) -> Dataset:
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, download=download)
    if dataset_name == "cifar100":
        return datasets.CIFAR100(root=root, train=True, download=download)
    if dataset_name == "stl10":
        # use train split (labeled)
        return datasets.STL10(root=root, split="train", download=download)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_transforms(dataset_name: str, img_size: int, augment: bool = True) -> transforms.Compose:
    dataset_name = dataset_name.lower()
    mean, std = DATASET_STATS[dataset_name]
    ops: List[torch.nn.Module] = []
    if augment:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(img_size, padding=4)])
    ops.extend(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)


class IndexedDataset(Dataset):
    """
    Wrap a dataset to always return its canonical index.
    """

    def __init__(self, base: Dataset, indices: Optional[Sequence[int]] = None, transform: Optional[Callable] = None):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, target = self.base[base_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target, int(base_idx)


def load_split_indices(path: str | Path) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "dataset": data["dataset"],
        "seed": data["seed"],
        "z": data["splits"]["Z"],
        "public": data["splits"]["Public"],
        "holdout": data["splits"]["Holdout"],
    }


def dataloader_from_indices(
    dataset_name: str,
    base_dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    img_size: int,
    augment: bool,
) -> DataLoader:
    transform = get_transforms(dataset_name, img_size, augment=augment)
    wrapped = IndexedDataset(base_dataset, indices=indices, transform=transform)

    def _worker_init(worker_id: int) -> None:
        # Ensure deterministic dataloader
        seed_offset = seed + worker_id
        torch.manual_seed(seed_offset)
        torch.cuda.manual_seed_all(seed_offset)

    generator = make_generator(seed)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init,
        generator=generator,
    )

