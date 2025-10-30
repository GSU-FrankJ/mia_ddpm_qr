"""Datasets for quantile regression on t-error scores."""

from __future__ import annotations

import pathlib
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split

from attacks.scores.compute_scores import SplitDataset, load_indices


class QuantileRegressionDataset(Dataset):
    """Pairs CIFAR-10 images with cached t-error scores.

    The dataset is built exclusively from the auxiliary split to avoid
    contaminating the membership evaluation sets.
    """

    def __init__(
        self,
        data_cfg: Dict,
        scores_path: pathlib.Path,
        limit: Optional[int] = None,
    ) -> None:
        aux_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["aux"]))
        root = pathlib.Path(data_cfg["dataset"]["root"])
        mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
        std = tuple(data_cfg["dataset"]["normalization"]["std"])
        base_dataset = SplitDataset(root, aux_indices, False, mean, std)
        cache = torch.load(scores_path, map_location="cpu")
        raw_scores = cache["scores"].float()
        if limit is not None and limit < len(base_dataset):
            subset_indices = list(range(limit))
            self.dataset = Subset(base_dataset, subset_indices)
            self.raw_scores = raw_scores[:limit]
        else:
            self.dataset = base_dataset
            self.raw_scores = raw_scores
        self.log_scores = torch.log1p(self.raw_scores.clamp_min(0))
        if len(self.dataset) != self.raw_scores.shape[0]:
            raise ValueError("Score cache size mismatch with auxiliary dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, _ = self.dataset[idx]
        return image, self.raw_scores[idx], self.log_scores[idx]


def train_val_split(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot split")
    val_size = max(1, int(len(dataset) * val_ratio)) if len(dataset) > 1 else 0
    val_size = min(len(dataset) - 1, val_size) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

