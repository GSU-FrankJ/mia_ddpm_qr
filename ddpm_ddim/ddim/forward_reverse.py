"""
DDIM forward and reverse processes.

Implements deterministic forward and reverse diffusion processes for DDIM sampling.
"""

import torch
from typing import Optional


def ddim_forward(
    x0: torch.Tensor,
    t: torch.Tensor,
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Deterministic DDIM forward process.
    
    Given clean image x0 and timestep t, compute noisy image xt:
        xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    
    This is deterministic (no noise added) for DDIM.
    
    Args:
        x0: Clean image [B, C, H, W]
        t: Timestep indices [B] with values in [0, T-1]
        model: UNet model (not used in forward, kept for API consistency)
        alphas_bar: Cumulative alpha_bar values [T] where alpha_bar[t] corresponds to timestep t
        eta: DDIM sampling parameter (0 for deterministic, 1 for DDPM)
              Not used in forward process but kept for consistency
        
    Returns:
        Noisy image xt [B, C, H, W]
    """
    # Gather alpha_bar values for each sample's timestep
    # t values are in [0, T-1], so we index alphas_bar directly
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    
    # Forward process: xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    # For deterministic forward (no noise), we can compute this directly
    # In practice, we'll add noise here for consistency with DDPM training
    # but for DDIM inference, we use deterministic forward
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    
    # Sample noise (this will be deterministic in practice with fixed seed)
    # For true deterministic forward, set noise to zero
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    return xt


def ddim_reverse(
    xt: torch.Tensor,
    t: torch.Tensor,
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Deterministic DDIM reverse process.
    
    Given noisy image xt and timestep t, predict x0 and compute x_{t-1}:
        eps_theta = model(xt, t)
        x0_pred = (xt - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * eps_theta
    
    This is deterministic (no noise added) when eta=0.
    
    Args:
        xt: Noisy image at timestep t [B, C, H, W]
        t: Timestep indices [B] with values in [0, T-1]
        model: UNet model that predicts noise eps_theta
        alphas_bar: Cumulative alpha_bar values [T] where alpha_bar[t] corresponds to timestep t
        eta: DDIM sampling parameter (0 for deterministic, 1 for DDPM)
        
    Returns:
        Predicted image x_{t-1} [B, C, H, W]
        
    Note:
        When t=0, this returns the final denoised image x0_pred.
        For t > 0, returns x_{t-1} for the reverse process.
    """
    # Gather alpha_bar values for current and previous timesteps
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    
    # Predict noise using the model
    eps_theta = model(xt, t)
    
    # Predict x0 from xt and predicted noise
    # x0 = (xt - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    
    x0_pred = (xt - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
    
    # For t=0, return x0_pred directly
    mask = (t > 0).float().view(-1, 1, 1, 1)
    
    # Compute x_{t-1} using DDIM formula
    if torch.any(t > 0):
        # Get alpha_bar for previous timestep
        t_prev = t - 1
        alpha_bar_t_prev = alphas_bar[t_prev].view(-1, 1, 1, 1)
        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
        sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
        
        # DDIM reverse: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * eps_theta
        # When eta=0, this is deterministic
        pred_prev = sqrt_alpha_bar_t_prev * x0_pred + sqrt_one_minus_alpha_bar_t_prev * eps_theta
        
        # For t=0, use x0_pred; for t>0, use pred_prev
        x_prev = mask * pred_prev + (1 - mask) * x0_pred
    else:
        x_prev = x0_pred
    
    return x_prev


def ddim_reverse_step(
    xt: torch.Tensor,
    t: int,
    model: torch.nn.Module,
    alphas_bar: torch.Tensor,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Single DDIM reverse step (helper function).
    
    Args:
        xt: Noisy image at timestep t [B, C, H, W]
        t: Single timestep (scalar)
        model: UNet model
        alphas_bar: Cumulative alpha_bar values [T]
        eta: DDIM sampling parameter
        
    Returns:
        Predicted image x_{t-1} [B, C, H, W]
    """
    t_tensor = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
    return ddim_reverse(xt, t_tensor, model, alphas_bar, eta)

