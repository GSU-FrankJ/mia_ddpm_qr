"""Reproducible CIFAR-10 membership splits.

This script materialises the deterministic splits described in the project
specification. The design deliberately separates the diffusion model training
subset (40k images) from the quantile-regression auxiliary data (10k test
images) to prevent leakage. The resulting JSON files reside in
`data/splits/` and are consumed by subsequent stages.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import torch
import torchvision
import yaml

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


def _load_config(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    LOGGER.debug("Loaded data config from %s", path)
    return cfg


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sample_indices(generator: torch.Generator, population: int, k: int) -> List[int]:
    perm = torch.randperm(population, generator=generator)
    return perm[:k].tolist()


def build_splits(cfg: Dict) -> Dict[str, List[int]]:
    """Construct the deterministic membership splits.

    Returns a dictionary with the following entries:
    - member_train: 40k indices from the CIFAR-10 training partition.
    - eval_in: 5k subset of `member_train` reserved for evaluation.
    - eval_out: 5k indices sampled from the 10k non-member pool.
    - aux: 10k indices from the CIFAR-10 test partition.

    The sampling uses a `torch.Generator` seeded from the config to ensure all
    downstream stages observe identical membership assignments.
    """

    generator = torch.Generator().manual_seed(cfg["splits"]["seed"])

    full_train = 50000
    member_train = _sample_indices(generator, full_train, 40000)

    aux = list(range(10000))

    member_tensor = torch.tensor(member_train)
    eval_in_perm = torch.randperm(member_tensor.size(0), generator=generator)
    eval_in = member_tensor[eval_in_perm[:5000]].tolist()

    member_set = set(member_train)
    non_member_pool = [idx for idx in range(full_train) if idx not in member_set]
    non_member_tensor = torch.tensor(non_member_pool)
    eval_out_perm = torch.randperm(non_member_tensor.size(0), generator=generator)
    eval_out = non_member_tensor[eval_out_perm[:5000]].tolist()

    LOGGER.info(
        "Constructed splits: member_train=%d, eval_in=%d, eval_out=%d, aux=%d",
        len(member_train),
        len(eval_in),
        len(eval_out),
        len(aux),
    )

    return {
        "member_train": member_train,
        "eval_in": eval_in,
        "eval_out": eval_out,
        "aux": aux,
    }


def save_splits(splits: Dict[str, List[int]], output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in splits.items():
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(indices, handle)
        LOGGER.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CIFAR-10 membership splits")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/data_cifar10.yaml"),
        help="Path to the CIFAR-10 data config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/splits"),
        help="Directory where JSON splits will be stored",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _set_seeds(cfg["splits"]["seed"])

    dataset_root = pathlib.Path(cfg["dataset"]["root"])
    dataset_root.mkdir(parents=True, exist_ok=True)

    torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=cfg["dataset"].get("download", True),
    )
    torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=False,
        download=cfg["dataset"].get("download", True),
    )

    splits = build_splits(cfg)
    save_splits(splits, args.output_dir)

    LOGGER.info(
        "Split complete. eval_in are part of member_train but held out from QR"
        " calibration to prevent attack leakage; eval_out never touches DDIM"
        " training, ensuring a clean negative set."
    )


if __name__ == "__main__":
    main()

