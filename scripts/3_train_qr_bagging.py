#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from attack_qr.engine.train_qr_bagging import QuantileTrainingConfig, train_bagging_ensemble
from ddpm.data.loader import load_split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quantile regression bagging ensemble.")
    parser.add_argument("--config", type=str, required=True, help="Attack YAML config.")
    parser.add_argument("--pairs", type=str, required=True, help="NPZ file with t-error pairs.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for models.")
    parser.add_argument("--split_json", type=str, default=None, help="Optional split JSON for public indices.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dataset_name = config.get("dataset", config.get("dataset_name"))
    if dataset_name is None:
        raise ValueError("Attack config must include 'dataset'.")
    bag_cfg = config.get("bagging", {})
    qr_cfg = config.get("qr", {})

    qt_config = QuantileTrainingConfig(
        lr=qr_cfg.get("lr", 1e-3),
        epochs=qr_cfg.get("epochs", 30),
        batch_size=qr_cfg.get("batch_size", 256),
        alpha_list=tuple(qr_cfg.get("alpha_list", [0.01, 0.001])),
        bootstrap=bag_cfg.get("bootstrap", True),
        M=bag_cfg.get("M", 16),
        seed=config.get("seed", 0),
    )

    pairs_path = Path(args.pairs)
    with np.load(pairs_path) as data:
        image_ids = data["image_id"]
    if args.split_json:
        public_indices = load_split_indices(args.split_json)["public"]
    else:
        public_indices = np.unique(image_ids).tolist()

    train_bagging_ensemble(
        npz_path=pairs_path,
        dataset_name=dataset_name,
        public_indices=public_indices,
        config=qt_config,
        out_dir=Path(args.out),
        img_size=config.get("img_size", 32),
        data_root=args.data_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()

