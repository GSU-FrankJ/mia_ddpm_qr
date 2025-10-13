import hashlib
import os
import random
from typing import Iterable, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """
    Create a Philox-based generator seeded with the provided seed.
    """
    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(seed)
    return generator


def stable_int_hash(parts: Iterable[int | str]) -> int:
    """
    Stable SHA1-based hash -> 64-bit integer.
    """
    hasher = hashlib.sha1()
    joined = "|".join(str(p) for p in parts)
    hasher.update(joined.encode("utf-8"))
    return int.from_bytes(hasher.digest()[:8], byteorder="big")


def philox_seed(dataset: str, sample_index: int, t: int, global_seed: int) -> int:
    """
    Derive a deterministic Philox seed tied to dataset/sample/timestep/global seed.
    """
    return stable_int_hash((dataset, sample_index, t, global_seed))


def timesteps_seed(dataset: str, sample_index: int, global_seed: int) -> int:
    return stable_int_hash((dataset, sample_index, "timesteps", global_seed))
