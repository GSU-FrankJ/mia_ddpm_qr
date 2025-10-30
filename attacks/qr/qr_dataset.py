"""Datasets for quantile regression on t-error scores."""

from __future__ import annotations

import pathlib
from typing import Dict

import torch
from torch.utils.data import Dataset, random_split

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
    ) -> None:
        aux_indices = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["aux"]))
        root = pathlib.Path(data_cfg["dataset"]["root"])
        mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
        std = tuple(data_cfg["dataset"]["normalization"]["std"])
        self.dataset = SplitDataset(root, aux_indices, False, mean, std)
        cache = torch.load(scores_path, map_location="cpu")
        self.scores = cache["scores"].float()
        if len(self.dataset) != self.scores.shape[0]:
            raise ValueError("Score cache size mismatch with auxiliary dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, _ = self.dataset[idx]
        score = self.scores[idx]
        return image, score


def train_val_split(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

