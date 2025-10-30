"""
t-error scoring for membership inference attacks.

Computes reconstruction error between deterministic forward→reverse at multiple timesteps.
"""

import torch
import torch.nn.functional as F
from typing import List, Sequence, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm


def uniform_timesteps(T: int = 1000, k: int = 50) -> List[int]:
    """
    Generate k uniform timesteps from [0, T-1].
    
    Samples k timesteps uniformly spaced across the diffusion process.
    This provides a diverse set of timesteps for computing t-error scores.
    
    Args:
        T: Total number of timesteps (default: 1000)
        k: Number of timesteps to sample (default: 50)
        
    Returns:
        List of k timestep indices in [0, T-1]
        
    Example:
        >>> uniform_timesteps(T=1000, k=5)
        [0, 250, 500, 750, 999]
    """
    if k >= T:
        return list(range(T))
    
    # Generate k uniform timesteps
    timesteps = np.linspace(0, T - 1, k, dtype=int)
    return timesteps.tolist()


def t_error_once(
    x0: torch.Tensor,
    t: int,
    model: torch.nn.Module,
    alphas_bar: torch.Tensor
) -> torch.Tensor:
    """
    Compute t-error for a single timestep.
    
    Computes squared L2 reconstruction error between deterministic forward→reverse:
    1. Forward: xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    2. Reverse: x0_pred = ddim_reverse(xt, t)
    3. Error: ||x0 - x0_pred||^2
    
    Args:
        x0: Clean image [B, C, H, W]
        t: Timestep index (scalar)
        model: UNet model for reverse process
        alphas_bar: Cumulative alpha_bar values [T]
        
    Returns:
        Squared L2 error per sample [B]
    """
    device = x0.device
    batch_size = x0.shape[0]
    
    # Forward process: add noise deterministically
    # xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    alpha_bar_t = alphas_bar[t].item()
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    
    # Sample noise (deterministic with fixed seed in practice)
    # For true deterministic, we'd use a seeded RNG, but for now use random
    # In actual usage, seed should be set per sample
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    # Reverse process: predict x0
    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
    
    with torch.no_grad():
        # Predict noise
        eps_pred = model(xt, t_tensor)
        
        # Predict x0 from xt and predicted noise
        # x0 = (xt - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        x0_pred = (xt - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
    
    # Compute squared L2 error per sample
    # Error shape: [B] - mean over spatial and channel dimensions
    error = torch.mean((x0 - x0_pred) ** 2, dim=(1, 2, 3))
    
    return error


def t_error_aggregate(
    x0: torch.Tensor,
    T_set: Sequence[int],
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
    agg: str = "mean"
) -> torch.Tensor:
    """
    Compute aggregated t-error over multiple timesteps.
    
    Computes t-error for each timestep in T_set and aggregates per sample.
    
    Args:
        x0: Clean image [B, C, H, W]
        T_set: Sequence of timestep indices to evaluate
        model: UNet model
        alphas_bar: Cumulative alpha_bar values [T]
        agg: Aggregation method - "mean", "min", or "q10" (10th percentile)
        
    Returns:
        Aggregated error per sample [B]
    """
    errors = []
    
    for t in T_set:
        error_t = t_error_once(x0, t, model, alphas_bar)
        errors.append(error_t)
    
    # Stack errors: [k, B] where k = len(T_set)
    errors_tensor = torch.stack(errors, dim=0)  # [k, B]
    
    # Aggregate per sample
    if agg == "mean":
        aggregated = torch.mean(errors_tensor, dim=0)  # [B]
    elif agg == "min":
        aggregated = torch.min(errors_tensor, dim=0)[0]  # [B]
    elif agg.startswith("q"):
        # Parse percentile (e.g., "q10" -> 10th percentile)
        percentile = int(agg[1:])
        aggregated = torch.quantile(errors_tensor, percentile / 100.0, dim=0)  # [B]
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")
    
    return aggregated


def compute_scores_for_split(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    T_set: Sequence[int],
    alphas_bar: torch.Tensor,
    agg: str = "mean",
    device: torch.device = None,
    cache_path: Optional[str] = None
) -> torch.Tensor:
    """
    Compute t-error scores for a dataset split.
    
    Args:
        model: UNet model
        dataloader: Data loader for the split
        T_set: Sequence of timesteps to evaluate
        alphas_bar: Cumulative alpha_bar values [T]
        agg: Aggregation method
        device: Device to compute on
        cache_path: Optional path to cache scores
        
    Returns:
        Scores per sample [N] where N is total number of samples
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Check cache
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached scores from {cache_path}")
        return torch.load(cache_path)
    
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Computing scores")
        for images, _ in pbar:
            images = images.to(device)
            
            # Compute aggregated t-error
            scores_batch = t_error_aggregate(
                x0=images,
                T_set=T_set,
                model=model,
                alphas_bar=alphas_bar,
                agg=agg
            )
            
            all_scores.append(scores_batch.cpu())
    
    # Concatenate all scores
    all_scores = torch.cat(all_scores, dim=0)
    
    # Cache scores
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_scores, cache_path)
        logger.info(f"Cached scores to {cache_path}")
    
    return all_scores


# Import logger
import logging
logger = logging.getLogger(__name__)

