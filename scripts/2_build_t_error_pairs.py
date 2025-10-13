#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from attack_qr.engine.build_pairs import build_t_error_pairs
from ddpm.data.loader import dataloader_from_indices, get_dataset, load_split_indices
from ddpm.engine.checkpoint_utils import load_ddpm_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic t-error pairs for Public split.")
    parser.add_argument("--config", type=str, required=True, help="Attack YAML config.")
    parser.add_argument("--split_json", type=str, required=True, help="Split JSON path.")
    parser.add_argument("--ddpm_ckpt", type=str, required=True, help="Trained DDPM checkpoint (best.pt).")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    public_cfg = config.get("public", {})
    dataset_name = config.get("dataset", config.get("dataset_name"))
    if dataset_name is None:
        raise ValueError("Attack config must include 'dataset'.")
    seed = config.get("seed", 0)
    K = public_cfg.get("K", 12)
    mode = public_cfg.get("mode", "eps")
    batch_size = public_cfg.get("batch_size", 32)

    split = load_split_indices(args.split_json)
    public_indices = split["public"]

    device = torch.device(args.device)
    model, schedule, metadata = load_ddpm_model(Path(args.ddpm_ckpt), device=device)

    loader = dataloader_from_indices(
        dataset_name=dataset_name,
        base_dataset=get_dataset(dataset_name, root=args.data_root, download=True),
        indices=public_indices,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        seed=seed,
        img_size=metadata.get("img_size", 32),
        augment=False,
    )

    out_path = Path(args.out)
    stats = build_t_error_pairs(
        model=model,
        schedule=schedule,
        dataloader=loader,
        dataset_name=dataset_name,
        global_seed=seed,
        K=K,
        mode=mode,
        out_path=out_path,
        device=device,
    )

    with out_path.with_suffix(".info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "attack_config": config,
                "ddpm_metadata": metadata,
                "pairs_stats": stats,
            },
            f,
            indent=2,
        )
    print(f"Saved pairs to {out_path}")


if __name__ == "__main__":
    main()
