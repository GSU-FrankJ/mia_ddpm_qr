"""CLI for training quantile regression bagging ensemble."""

from __future__ import annotations

import argparse
import pathlib

import sys

import torch
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from attacks.qr.bagging import BagOfQuantiles


LOGGER = get_winston_logger(__name__)


def load_yaml(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quantile regression ensemble")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/attack_qr.yaml"))
    parser.add_argument("--data-config", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--fastdev", action="store_true", help="Enable fast development mode")
    args = parser.parse_args()

    attack_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    base_cfg = dict(attack_cfg["train"])
    base_cfg["tau_values"] = attack_cfg["bagging"]["tau_values"]
    limit = None
    B = attack_cfg["bagging"]["B"]
    bootstrap_ratio = attack_cfg["bagging"]["bootstrap_ratio"]

    if args.fastdev:
        base_cfg["epochs"] = min(base_cfg.get("epochs", 1), 2)
        base_cfg["batch_size"] = min(base_cfg.get("batch_size", 64), 64)
        base_cfg["val_ratio"] = base_cfg.get("val_ratio", 0.1)
        limit = 512
        B = min(B, 5)
        bootstrap_ratio = min(bootstrap_ratio, 0.5)
        LOGGER.warning("Fastdev mode: limiting dataset to %s samples, B=%s", limit, B)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble = BagOfQuantiles(
        base_cfg=base_cfg,
        B=B,
        bootstrap_ratio=bootstrap_ratio,
        seed=attack_cfg["seed"],
        device=device,
    )

    scores_path = pathlib.Path(attack_cfg["t_error"]["cache_dir"]) / "aux.pt"
    LOGGER.info("Training ensemble using scores: %s", scores_path)
    ensemble.fit(scores_path, data_cfg, limit=limit)

    output_dir = pathlib.Path(attack_cfg["logging"]["output_dir"]) / "ensembles"
    ensemble_path = output_dir / "bagging.pt"
    ensemble.save(ensemble_path)


if __name__ == "__main__":
    main()


