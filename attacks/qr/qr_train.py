"""
Training script for quantile regression models.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import logging

from attacks.qr.qr_models import SmallCNNQuantile, pinball_loss
from attacks.qr.qr_dataset import QuantilePairsDataset

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    tau: float,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Quantile regression model
        dataloader: Data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        tau: Quantile level
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Training (tau={tau})")
    for images, scores in pbar:
        images = images.to(device)
        scores = scores.to(device)
        
        # Forward pass
        pred_thresholds = model(images)  # [B]
        
        # Compute pinball loss
        loss = pinball_loss(pred_thresholds, scores, tau).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    tau: float,
    device: torch.device
) -> float:
    """
    Validate model.
    
    Args:
        model: Quantile regression model
        dataloader: Validation data loader
        tau: Quantile level
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, scores in dataloader:
            images = images.to(device)
            scores = scores.to(device)
            
            pred_thresholds = model(images)
            loss = pinball_loss(pred_thresholds, scores, tau).mean()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_quantile_regressor(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tau: float,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path
) -> nn.Module:
    """
    Train a single quantile regression model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        tau: Quantile level
        config: Training configuration
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Trained model
    """
    # Create model
    model_config = config["qr"]["model"]
    model = SmallCNNQuantile(
        in_channels=3,
        channels=model_config["channels"],
        kernel_size=model_config["kernel_size"],
        stride=model_config["stride"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Setup optimizer
    train_config = config["qr"]["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        betas=tuple(train_config["betas"])
    )
    
    # Setup learning rate scheduler
    if train_config["lr_schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config["epochs"],
            eta_min=train_config["lr"] * 0.01
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    epochs = train_config["epochs"]
    
    logger.info(f"Training quantile regressor for tau={tau}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, tau, device)
        val_loss = validate(model, val_loader, tau, device)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )
        
        # Early stopping
        if train_config["early_stopping"]["enabled"]:
            if val_loss < best_val_loss - train_config["early_stopping"]["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= train_config["early_stopping"]["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Load best model
    if train_config["early_stopping"]["enabled"]:
        model.load_state_dict(torch.load(checkpoint_dir / "best.pt"))
    
    return model


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train quantile regression models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to attack config YAML"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load aux indices
    aux_json = config["qr"]["data"]["aux_indices_json"]
    with open(aux_json, "r") as f:
        aux_indices = json.load(f)
    
    # Load aux scores
    aux_scores_path = config["t_error"]["aux_cache"]
    aux_scores = torch.load(aux_scores_path)
    
    logger.info(f"Loaded {len(aux_indices)} aux samples with scores")
    
    # Split aux into train/val
    val_ratio = config["qr"]["train"]["val_ratio"]
    n_val = int(len(aux_indices) * val_ratio)
    n_train = len(aux_indices) - n_val
    
    train_indices = aux_indices[:n_train]
    val_indices = aux_indices[n_train:]
    train_scores = aux_scores[:n_train]
    val_scores = aux_scores[n_train:]
    
    logger.info(f"Train: {n_train} samples, Val: {n_val} samples")
    
    # Create datasets
    image_root = config["qr"]["data"]["image_root"]
    train_dataset = QuantilePairsDataset(train_indices, image_root, train_scores)
    val_dataset = QuantilePairsDataset(val_indices, image_root, val_scores)
    
    # Create data loaders
    batch_size = config["qr"]["train"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train models for each quantile
    quantiles = config["qr"]["quantiles"]
    output_dir = Path(config["qr"]["output"]["model_dir"])
    
    for tau in quantiles:
        logger.info(f"\n=== Training model for tau={tau} ===")
        
        checkpoint_dir = output_dir / f"tau_{tau}"
        
        model = train_quantile_regressor(
            train_loader=train_loader,
            val_loader=val_loader,
            tau=tau,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save final model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "final.pt")
        logger.info(f"Saved model to {checkpoint_dir}/final.pt")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()

