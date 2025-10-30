"""
Dataset for quantile regression training.

Creates (image, score) pairs from aux split for QR training.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from typing import List


class QuantilePairsDataset(Dataset):
    """
    Dataset of (image, score) pairs for quantile regression.
    
    Loads images from aux split and pairs them with pre-computed t-error scores.
    
    Args:
        image_indices: List of image indices in the dataset
        image_root: Root directory for CIFAR-10 images
        scores: Pre-computed t-error scores [N] matching image_indices
        normalize: Whether to apply normalization
    """
    def __init__(
        self,
        image_indices: List[int],
        image_root: str,
        scores: torch.Tensor,
        normalize: bool = True
    ):
        self.image_indices = image_indices
        self.scores = scores
        
        # Load CIFAR-10 dataset
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            )
        transform = transforms.Compose(transform_list)
        
        # Load full CIFAR-10 test set (aux uses test set)
        self.dataset = torchvision.datasets.CIFAR10(
            root=image_root, train=False, download=True, transform=transform
        )
        
        assert len(image_indices) == len(scores), \
            f"Mismatch: {len(image_indices)} indices vs {len(scores)} scores"
    
    def __len__(self) -> int:
        return len(self.image_indices)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get (image, score) pair.
        
        Args:
            idx: Index in dataset
            
        Returns:
            Tuple of (image [3, 32, 32], score [scalar])
        """
        image_idx = self.image_indices[idx]
        image, _ = self.dataset[image_idx]
        score = self.scores[idx].item()
        
        return image, score

