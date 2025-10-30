"""
Data splitting module for reproducible CIFAR-10 splits.

This module implements a strict split protocol to ensure no data leakage:
- member_train (40k): Used to train DDIM
- eval_in (5k): Positive samples from member_train (excluded from QR training)
- eval_out (5k): Negative samples from remaining 10k train images (never used by DDIM)
- aux (10k): Full test set for QR training/calibration only
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms

# Set random seeds for reproducibility
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cifar10_indices(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset and return train/test indices.
    
    Args:
        data_root: Root directory for CIFAR-10 data
        
    Returns:
        Tuple of (train_indices, test_indices) where indices are class-stratified
    """
    # Load CIFAR-10 dataset (downloads if needed)
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    
    # Get all indices
    train_indices = np.arange(len(train_dataset))
    test_indices = np.arange(len(test_dataset))
    
    return train_indices, test_indices


def create_splits(
    member_train_size: int = 40000,
    eval_in_size: int = 5000,
    eval_out_size: int = 5000,
    aux_size: int = 10000,
    seed: int = 20251030,
    data_root: str = "data/cifar10"
) -> Dict[str, List[int]]:
    """
    Create reproducible data splits with no leakage.
    
    Split protocol:
    1. Split 50k train images into:
       - member_train (40k): Used to train DDIM
       - remaining_train (10k): Never used by DDIM
    2. From member_train, sample eval_in (5k) for positive evaluation
    3. From remaining_train, sample eval_out (5k) for negative evaluation
    4. Use full test set (10k) as aux for QR training
    
    Args:
        member_train_size: Size of member_train split
        eval_in_size: Size of eval_in (positive samples)
        eval_out_size: Size of eval_out (negative samples)
        aux_size: Size of aux split (should be 10k for full test set)
        seed: Random seed for reproducibility
        data_root: Root directory for CIFAR-10 data
        
    Returns:
        Dictionary mapping split names to lists of indices
    """
    set_seed(seed)
    
    # Load dataset indices
    train_indices, test_indices = load_cifar10_indices(data_root)
    
    # Step 1: Split train set into member_train (40k) and remaining_train (10k)
    # Use stratified splitting to maintain class distribution
    # We'll use a simple random split since we don't have labels here
    # but we maintain randomness with seed
    np.random.seed(seed)
    np.random.shuffle(train_indices)
    
    member_train_indices = train_indices[:member_train_size].tolist()
    remaining_train_indices = train_indices[member_train_size:].tolist()
    
    # Step 2: Sample eval_in from member_train (5k)
    # These are positive samples but excluded from QR training
    np.random.seed(seed + 1)  # Different seed for this split
    eval_in_indices = np.random.choice(
        member_train_indices, size=eval_in_size, replace=False
    ).tolist()
    
    # Step 3: Sample eval_out from remaining_train (5k)
    # These are negative samples (never seen by DDIM)
    np.random.seed(seed + 2)  # Different seed for this split
    eval_out_indices = np.random.choice(
        remaining_train_indices, size=eval_out_size, replace=False
    ).tolist()
    
    # Step 4: Use full test set as aux (10k)
    # This is used ONLY for QR training/calibration
    aux_indices = test_indices[:aux_size].tolist()
    
    return {
        "aux": sorted(aux_indices),
        "eval_in": sorted(eval_in_indices),
        "eval_out": sorted(eval_out_indices),
        "member_train": sorted(member_train_indices),
        "remaining_train": sorted(remaining_train_indices)
    }


def save_splits(splits: Dict[str, List[int]], output_dir: str) -> None:
    """
    Save splits to JSON files.
    
    Args:
        splits: Dictionary mapping split names to index lists
        output_dir: Directory to save JSON files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each split as a separate JSON file
    for split_name, indices in splits.items():
        if split_name in ["aux", "eval_in", "eval_out"]:
            output_file = output_path / f"{split_name}.json"
            with open(output_file, "w") as f:
                json.dump(indices, f, indent=2)
            print(f"Saved {split_name}: {len(indices)} indices to {output_file}")


def main() -> None:
    """Main entry point for data splitting script."""
    parser = argparse.ArgumentParser(
        description="Create reproducible CIFAR-10 splits for QR-MIA"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/cifar10",
        help="Root directory for CIFAR-10 dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Output directory for split JSON files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20251030,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--member-train-size",
        type=int,
        default=40000,
        help="Size of member_train split"
    )
    parser.add_argument(
        "--eval-in-size",
        type=int,
        default=5000,
        help="Size of eval_in split"
    )
    parser.add_argument(
        "--eval-out-size",
        type=int,
        default=5000,
        help="Size of eval_out split"
    )
    parser.add_argument(
        "--aux-size",
        type=int,
        default=10000,
        help="Size of aux split"
    )
    
    args = parser.parse_args()
    
    print("Creating CIFAR-10 splits with no leakage...")
    print(f"Seed: {args.seed}")
    print(f"member_train: {args.member_train_size}")
    print(f"eval_in: {args.eval_in_size}")
    print(f"eval_out: {args.eval_out_size}")
    print(f"aux: {args.aux_size}")
    
    splits = create_splits(
        member_train_size=args.member_train_size,
        eval_in_size=args.eval_in_size,
        eval_out_size=args.eval_out_size,
        aux_size=args.aux_size,
        seed=args.seed,
        data_root=args.data_root
    )
    
    save_splits(splits, args.output_dir)
    
    print("\nSplit summary:")
    for split_name, indices in splits.items():
        print(f"  {split_name}: {len(indices)} indices")
    
    print(f"\nAll splits saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

