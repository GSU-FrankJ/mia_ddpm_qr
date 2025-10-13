#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from attack_qr.utils.seeding import seed_everything
from ddpm.data.loader import get_dataset
from ddpm.data.split import stratified_split


def compute_class_distribution(indices: Sequence[int], labels: Sequence[int]) -> Dict[int, int]:
    counts = defaultdict(int)
    for idx in indices:
        counts[int(labels[idx])] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits for QR-MIA pipeline.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., cifar10).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path for splits.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory.")
    args = parser.parse_args()

    seed_everything(args.seed)

    dataset = get_dataset(args.dataset, root=args.data_root, download=True)
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise AttributeError("Dataset does not expose labels for stratification.")

    splits = stratified_split(labels, seed=args.seed)

    info = {
        "dataset": args.dataset,
        "seed": args.seed,
        "num_samples": len(dataset),
        "splits": splits,
        "class_distribution": {
            split: compute_class_distribution(indices, labels) for split, indices in splits.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote splits to {out_path}")


if __name__ == "__main__":
    main()
