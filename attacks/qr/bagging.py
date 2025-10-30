"""
Bagging ensemble for quantile regression.

Implements bootstrap aggregation (bagging) of quantile regression models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

from attacks.qr.qr_models import SmallCNNQuantile

logger = logging.getLogger(__name__)


class BagOfQuantiles:
    """
    Bagging ensemble of quantile regression models.
    
    Trains B models on bootstrap samples of aux data, then aggregates predictions
    using majority vote or average threshold.
    
    Args:
        base_cfg: Configuration dictionary for base model
        B: Number of bootstrap models (default: 50)
        bootstrap_ratio: Fraction of data to sample per model (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
    """
    def __init__(
        self,
        base_cfg: dict,
        B: int = 50,
        bootstrap_ratio: float = 0.8,
        seed: int = 42
    ):
        self.base_cfg = base_cfg
        self.B = B
        self.bootstrap_ratio = bootstrap_ratio
        self.seed = seed
        
        self.models: List[nn.Module] = []
        self.bootstrap_indices: List[List[int]] = []
        
        # Set random seed
        np.random.seed(seed)
    
    def fit(
        self,
        aux_loader: DataLoader,
        tau: float,
        device: torch.device,
        train_fn: callable
    ) -> None:
        """
        Fit B models on bootstrap samples.
        
        Args:
            aux_loader: DataLoader for aux dataset
            tau: Quantile level
            device: Device to train on
            train_fn: Function to train a single model (should return trained model)
        """
        # Get full dataset and indices
        full_dataset = aux_loader.dataset
        n_samples = len(full_dataset)
        n_bootstrap = int(n_samples * self.bootstrap_ratio)
        
        logger.info(f"Fitting {self.B} models with bootstrap size {n_bootstrap}/{n_samples}")
        
        for b in range(self.B):
            # Sample bootstrap indices
            bootstrap_idx = np.random.choice(n_samples, size=n_bootstrap, replace=True)
            self.bootstrap_indices.append(bootstrap_idx.tolist())
            
            # Create bootstrap dataset
            bootstrap_dataset = Subset(full_dataset, bootstrap_idx)
            bootstrap_loader = DataLoader(
                bootstrap_dataset,
                batch_size=aux_loader.batch_size,
                shuffle=True
            )
            
            # Train model
            logger.info(f"Training model {b+1}/{self.B}")
            model = train_fn(bootstrap_loader, tau, device, model_seed=self.seed + b)
            
            self.models.append(model)
    
    def predict_thresholds(
        self,
        images: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Predict thresholds for images using all models.
        
        Args:
            images: Input images [N, C, H, W]
            device: Device to compute on
            
        Returns:
            Predicted thresholds [B, N] where B is number of models
        """
        thresholds = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                images_gpu = images.to(device)
                pred = model(images_gpu).cpu()  # [N]
                thresholds.append(pred)
        
        return torch.stack(thresholds, dim=0)  # [B, N]
    
    def decision(
        self,
        scores: torch.Tensor,
        imgs: torch.Tensor,
        tau: float,
        device: torch.device,
        method: str = "majority"
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Make membership decisions using bagging ensemble.
        
        Args:
            scores: t-error scores [N]
            imgs: Input images [N, C, H, W]
            tau: Quantile level (used for thresholding)
            device: Device to compute on
            method: Aggregation method - "majority" or "average"
            
        Returns:
            Tuple of:
            - decisions [N]: Binary decisions (1=IN, 0=OUT)
            - diagnostics: Dictionary with per-model thresholds and other info
        """
        # Predict thresholds for all models
        thresholds = self.predict_thresholds(imgs, device)  # [B, N]
        
        # Make per-model decisions: score < threshold -> IN (1)
        per_model_decisions = (scores.unsqueeze(0) < thresholds).long()  # [B, N]
        
        if method == "majority":
            # Majority vote: decision = 1 if majority of models say IN
            decisions = (per_model_decisions.sum(dim=0) > self.B / 2).long()  # [N]
        elif method == "average":
            # Average thresholds, then single cutoff
            avg_threshold = thresholds.mean(dim=0)  # [N]
            decisions = (scores < avg_threshold).long()  # [N]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        diagnostics = {
            "per_model_thresholds": thresholds.tolist(),  # [B, N]
            "per_model_decisions": per_model_decisions.tolist(),  # [B, N]
            "decisions": decisions.tolist(),  # [N]
            "scores": scores.tolist(),  # [N]
            "method": method,
            "tau": tau
        }
        
        return decisions, diagnostics
    
    def save(self, path: Path) -> None:
        """Save ensemble to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model states
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")
        
        # Save metadata
        metadata = {
            "B": self.B,
            "bootstrap_ratio": self.bootstrap_ratio,
            "seed": self.seed,
            "bootstrap_indices": self.bootstrap_indices
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved ensemble to {path}")
    
    def load(self, path: Path, model_cfg: dict, device: torch.device) -> None:
        """Load ensemble from disk."""
        # Load metadata
        with open(path / "manifest.json", "r") as f:
            metadata = json.load(f)
        
        self.B = metadata["B"]
        self.bootstrap_ratio = metadata["bootstrap_ratio"]
        self.seed = metadata["seed"]
        self.bootstrap_indices = metadata["bootstrap_indices"]
        
        # Load models
        self.models = []
        for i in range(self.B):
            model = SmallCNNQuantile(
                in_channels=3,
                channels=model_cfg["channels"],
                kernel_size=model_cfg["kernel_size"],
                stride=model_cfg["stride"],
                dropout=model_cfg["dropout"]
            )
            model.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
            model.to(device)
            self.models.append(model)
        
        logger.info(f"Loaded ensemble from {path}")

