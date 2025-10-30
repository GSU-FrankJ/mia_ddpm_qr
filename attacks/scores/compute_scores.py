"""CLI to compute t-error scores for auxiliary and evaluation splits."""

from __future__ import annotations

import argparse
import pathlib
import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import yaml

from mia_logging import get_winston_logger
from ddpm_ddim.models import build_unet
from ddpm_ddim.schedulers import build_cosine_schedule
from attacks.scores import t_error_aggregate, uniform_timesteps


LOGGER = get_winston_logger(__name__)


class SplitDataset(Dataset):
    """Construct a dataset restricted to specific indices."""

    def __init__(
        self,
        root: pathlib.Path,
        indices: List[int],
        train: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        base = CIFAR10(root=str(root), train=train, download=False, transform=transform)
        self.subset = Subset(base, indices)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        return self.subset[idx]


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_indices(path: pathlib.Path) -> List[int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_checkpoint(root: pathlib.Path) -> pathlib.Path:
    candidates = sorted(
        [p for p in root.glob("ckpt_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {root}")
    return candidates[-1]


def load_model_from_checkpoint(
    model_cfg_path: pathlib.Path,
    checkpoint_root: pathlib.Path,
    prefer_ema: bool,
    device: torch.device,
) -> torch.nn.Module:
    model_cfg = load_yaml(model_cfg_path)
    model = build_unet(model_cfg["model"])
    ckpt_dir = find_latest_checkpoint(checkpoint_root)
    ckpt_path = ckpt_dir / ("ema.ckpt" if prefer_ema else "model.ckpt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    LOGGER.info("Loaded model weights from %s", ckpt_path)
    return model


def build_loader(
    data_cfg: Dict,
    split_name: str,
    indices: List[int],
    train_flag: bool,
    batch_size: int,
    fastdev: bool,
) -> DataLoader:
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])
    if fastdev:
        indices = indices[:1024]
    dataset = SplitDataset(root, indices, train_flag, mean, std)
    LOGGER.info("%s dataset size=%d", split_name, len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["dataset"].get("num_workers", 8),
        pin_memory=True,
    )


def compute_split_scores(
    name: str,
    loader: DataLoader,
    model: torch.nn.Module,
    timesteps: List[int],
    alphas_bar: torch.Tensor,
    agg: str,
    device: torch.device,
) -> torch.Tensor:
    scores = []
    for batch, _ in tqdm(loader, desc=f"scores-{name}"):
        images = batch.to(device)
        batch_scores = t_error_aggregate(images, timesteps, model, alphas_bar, agg)
        scores.append(batch_scores.cpu())
    concatenated = torch.cat(scores)
    LOGGER.info("%s scores shape=%s", name, tuple(concatenated.shape))
    return concatenated


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute t-error scores")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/attack_qr.yaml"))
    parser.add_argument("--data-config", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fastdev", action="store_true", help="Limit dataset and timesteps for dev runs")
    args = parser.parse_args()

    attack_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    timesteps = uniform_timesteps(
        attack_cfg["t_error"]["T"],
        5 if args.fastdev else attack_cfg["t_error"]["k_uniform"],
    )
    LOGGER.info("Using timesteps: %s", timesteps[:5] if len(timesteps) > 5 else timesteps)

    _betas, alphas_bar = build_cosine_schedule(attack_cfg["t_error"]["T"])
    alphas_bar = alphas_bar.to(device)

    model = load_model_from_checkpoint(
        pathlib.Path(attack_cfg["model"]["config"]),
        pathlib.Path(attack_cfg["model"]["checkpoint_root"]),
        attack_cfg["model"].get("prefer_ema", True),
        device,
    )

    splits_dir = pathlib.Path(data_cfg["splits"]["paths"]["aux"]).parent
    indices = {
        "aux": load_indices(pathlib.Path(data_cfg["splits"]["paths"]["aux"])),
        "eval_in": load_indices(pathlib.Path(data_cfg["splits"]["paths"]["eval_in"])),
        "eval_out": load_indices(pathlib.Path(data_cfg["splits"]["paths"]["eval_out"])),
    }

    loaders = {
        "aux": build_loader(data_cfg, "aux", indices["aux"], False, attack_cfg["train"]["batch_size"], args.fastdev),
        "eval_in": build_loader(data_cfg, "eval_in", indices["eval_in"], True, attack_cfg["train"]["batch_size"], args.fastdev),
        "eval_out": build_loader(data_cfg, "eval_out", indices["eval_out"], True, attack_cfg["train"]["batch_size"], args.fastdev),
    }

    cache_dir = pathlib.Path(attack_cfg["t_error"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, loader in loaders.items():
        scores = compute_split_scores(
            name,
            loader,
            model,
            timesteps,
            alphas_bar,
            attack_cfg["t_error"]["aggregate"],
            device,
        )
        out_path = cache_dir / f"{name}.pt"
        torch.save(
            {
                "scores": scores,
                "timesteps": timesteps,
                "aggregate": attack_cfg["t_error"]["aggregate"],
            },
            out_path,
        )
        results[name] = out_path
        LOGGER.info("Saved %s scores to %s", name, out_path)


if __name__ == "__main__":
    main()

