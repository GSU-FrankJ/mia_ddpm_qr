"""
DDIM training script for CIFAR-10.

Trains a DDIM model from scratch on CIFAR-10 with:
- AdamW optimizer
- EMA
- Mixed precision training
- Gradient clipping
- Checkpointing
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml
from rich.console import Console
from rich.logging import RichHandler
import logging

from ddpm_ddim.models.unet import UNet
from ddpm_ddim.schedulers.betas import get_schedule


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"


class EMAModel:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Maintains a copy of model parameters with exponential moving average:
    ema_param = decay * ema_param + (1 - decay) * model_param
    
    Args:
        model: Model to track
        decay: EMA decay rate (default: 0.9999)
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self) -> None:
        """Register all model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def load_cifar10_split(
    data_root: str,
    split_json: str,
    normalize: bool = True
) -> torch.utils.data.Dataset:
    """
    Load CIFAR-10 dataset with specific split indices.
    
    Args:
        data_root: Root directory for CIFAR-10 data
        split_json: Path to JSON file with indices
        normalize: Whether to apply normalization
        
    Returns:
        Subset of CIFAR-10 dataset
    """
    import json
    
    # Load indices
    with open(split_json, "r") as f:
        indices = json.load(f)
    
    # Load full CIFAR-10 train set
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        )
    transform = transforms.Compose(transform_list)
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    
    # Create subset
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    
    return subset


def train_epoch(
    model: nn.Module,
    ema_model: Optional[EMAModel],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    alphas_bar: torch.Tensor,
    T: int,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: UNet model
        ema_model: EMA model (optional)
        dataloader: Data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        scaler: Gradient scaler for AMP
        alphas_bar: Alpha bar schedule [T]
        T: Total timesteps
        device: Device to train on
        grad_clip: Gradient clipping value
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample timesteps uniformly
        t = torch.randint(0, T, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Get alpha_bar for each timestep
        alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
        
        # Forward process: add noise to images
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        noisy_images = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise
        
        # Forward pass
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred_noise = model(noisy_images, t)
            loss = F.mse_loss(pred_noise, noise)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        if ema_model is not None:
            ema_model.update()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
    
    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    ema_model: Optional[EMAModel],
    optimizer: torch.optim.Optimizer,
    iteration: int,
    checkpoint_dir: Path,
    metadata: Dict
) -> None:
    """
    Save checkpoint.
    
    Args:
        model: Model to save
        ema_model: EMA model (optional)
        optimizer: Optimizer state
        iteration: Current iteration
        checkpoint_dir: Directory to save checkpoint
        metadata: Metadata to save
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    model_path = checkpoint_dir / f"model.ckpt"
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, model_path)
    
    # Save EMA checkpoint
    if ema_model is not None:
        ema_path = checkpoint_dir / f"ema.ckpt"
        torch.save({
            "iteration": iteration,
            "ema_state_dict": ema_model.shadow,
        }, ema_path)
    
    # Save metadata
    metadata_path = checkpoint_dir / "run.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved checkpoint at iteration {iteration}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DDIM on CIFAR-10")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["main", "fastdev"],
        default="main",
        help="Training mode: main (400k iters) or fastdev (100k iters)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data config YAML"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on"
    )
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)
    
    # Set seed
    seed = model_config.get("seed", 20251030)
    set_seed(seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get training iterations
    if args.mode == "main":
        num_iterations = model_config["training"]["num_iterations"]
    else:
        num_iterations = model_config["training"]["fastdev_iterations"]
    
    logger.info(f"Training mode: {args.mode} ({num_iterations} iterations)")
    
    # Load dataset
    split_json = model_config["paths"]["data_split_json"]
    data_root = data_config["dataset"]["root"]
    
    logger.info(f"Loading dataset from {data_root}")
    logger.info(f"Using split: {split_json}")
    
    dataset = load_cifar10_split(
        data_root=data_root,
        split_json=split_json,
        normalize=True
    )
    
    # Create data loader
    batch_size = model_config["training"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config["dataloader"]["num_workers"],
        pin_memory=data_config["dataloader"]["pin_memory"]
    )
    
    # Compute number of epochs
    num_epochs = (num_iterations + len(dataloader) - 1) // len(dataloader)
    logger.info(f"Training for {num_epochs} epochs ({num_iterations} iterations)")
    
    # Create model
    unet_config = model_config["model"]["unet"]
    model = UNet(
        in_channels=unet_config["in_channels"],
        out_channels=unet_config["out_channels"],
        model_channels=unet_config["model_channels"],
        channel_mult=tuple(unet_config["channel_mult"]),
        num_res_blocks=unet_config["num_res_blocks"],
        attention_resolutions=tuple(unet_config["attention_resolutions"]),
        dropout=unet_config["dropout"],
        use_scale_shift_norm=unet_config["use_scale_shift_norm"]
    ).to(device)
    
    logger.info(f"Created UNet with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup optimizer
    opt_config = model_config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config["lr"],
        betas=tuple(opt_config["betas"]),
        weight_decay=opt_config.get("weight_decay", 0.01)
    )
    
    # Setup EMA
    ema_model = None
    if model_config["training"]["ema"]["enabled"]:
        ema_model = EMAModel(model, decay=model_config["training"]["ema"]["decay"])
        logger.info("EMA enabled")
    
    # Setup AMP
    scaler = torch.cuda.amp.GradScaler() if model_config["training"]["amp"]["enabled"] else None
    
    # Get noise schedule
    schedule_config = model_config["schedule"]
    _, alphas_bar = get_schedule(
        schedule_type=schedule_config["type"],
        T=schedule_config["T"],
        beta_start=schedule_config["beta_start"],
        beta_end=schedule_config["beta_end"]
    )
    alphas_bar = alphas_bar.to(device)
    T = schedule_config["T"]
    
    # Setup checkpoint directory
    checkpoint_dir = Path(model_config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    iteration = 0
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            alphas_bar=alphas_bar,
            T=T,
            device=device,
            grad_clip=model_config["training"]["grad_clip"]
        )
        
        iteration += len(dataloader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Iteration = {iteration}")
        
        # Save checkpoint
        if (iteration % model_config["training"]["checkpoint"]["save_every"] == 0) or (epoch == num_epochs - 1):
            metadata = {
                "git_hash": get_git_hash(),
                "seed": seed,
                "iteration": iteration,
                "epoch": epoch + 1,
                "config": model_config,
                "data_config": data_config
            }
            save_checkpoint(model, ema_model, optimizer, iteration, checkpoint_dir, metadata)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

