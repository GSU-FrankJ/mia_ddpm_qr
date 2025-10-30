"""
Noise schedule utilities for DDIM.

Implements cosine noise schedule for diffusion models.
"""

import torch
import numpy as np
from typing import Tuple


def cosine_beta_schedule(
    T: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> torch.Tensor:
    """
    Generate cosine noise schedule.
    
    Uses cosine schedule: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    where alpha_bar_t follows a cosine schedule.
    
    Args:
        T: Total number of timesteps
        beta_start: Starting beta value (unused in cosine schedule, kept for compatibility)
        beta_end: Ending beta value (unused in cosine schedule, kept for compatibility)
        
    Returns:
        Beta schedule tensor [T]
    """
    # Cosine schedule: alpha_bar_t = cos^2(pi/2 * t/T)
    # This gives smoother transitions compared to linear schedule
    s = 0.008  # Small offset for numerical stability
    steps = torch.arange(T + 1, dtype=torch.float32)
    alpha_bar = torch.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # Normalize
    
    # Compute beta from alpha_bar: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    alphas = alpha_bar[1:] / alpha_bar[:-1]
    betas = 1 - alphas
    
    return betas


def linear_beta_schedule(
    T: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> torch.Tensor:
    """
    Generate linear noise schedule (alternative to cosine).
    
    Args:
        T: Total number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Beta schedule tensor [T]
    """
    return torch.linspace(beta_start, beta_end, T)


def compute_alphas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute alpha and alpha_bar from beta schedule.
    
    Alpha and alpha_bar are used in the diffusion forward process:
    - alpha_t = 1 - beta_t
    - alpha_bar_t = prod_{s=1}^t alpha_s
    
    Args:
        betas: Beta schedule [T]
        
    Returns:
        Tuple of (alphas, alpha_bar) where:
        - alphas [T]: alpha_t = 1 - beta_t
        - alpha_bar [T]: Cumulative product of alphas
    """
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar


def get_schedule(
    schedule_type: str = "cosine",
    T: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get noise schedule and compute alpha_bar.
    
    Args:
        schedule_type: Type of schedule ("cosine" or "linear")
        T: Total number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Tuple of (betas, alpha_bar) where:
        - betas [T]: Noise schedule
        - alpha_bar [T]: Cumulative product of alphas
    """
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T, beta_start, beta_end)
    elif schedule_type == "linear":
        betas = linear_beta_schedule(T, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    _, alpha_bar = compute_alphas(betas)
    return betas, alpha_bar

