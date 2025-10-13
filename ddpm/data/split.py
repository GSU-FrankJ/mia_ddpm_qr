from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def stratified_split(labels: Sequence[int], seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    split_indices = {"Z": [], "Public": [], "Holdout": []}
    classes = np.unique(labels)
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        total = len(cls_indices)
        z_count = int(round(0.5 * total))
        remaining = total - z_count
        public_count = remaining // 2
        holdout_count = remaining - public_count
        split_indices["Z"].extend(cls_indices[:z_count].tolist())
        split_indices["Public"].extend(cls_indices[z_count : z_count + public_count].tolist())
        split_indices["Holdout"].extend(cls_indices[z_count + public_count :].tolist())
    for key in split_indices:
        split_indices[key].sort()
    return split_indices
