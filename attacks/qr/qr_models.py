"""
Quantile Regression models for membership inference.

Implements SmallCNNQuantile model that predicts per-sample thresholds using quantile regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SmallCNNQuantile(nn.Module):
    """
    Small CNN for quantile regression on image scores.
    
    Architecture:
    - 3 convolutional blocks (Conv-BN-ReLU) with channel progression
    - Global Average Pooling
    - Fully connected layer -> scalar threshold
    
    Args:
        in_channels: Input channels (3 for RGB)
        channels: List of channel counts for each conv block [64, 128, 128]
        kernel_size: Kernel size for convolutions (default: 3)
        stride: Stride for convolutions (default: 2)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: list = [64, 128, 128],
        kernel_size: int = 3,
        stride: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        
        # First conv block
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        
        # Second conv block
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        
        # Third conv block
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final fully connected layer -> scalar threshold
        self.fc = nn.Linear(channels[2], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Predicted threshold [B, 1]
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # [B, C]
        
        # Dropout
        x = self.dropout(x)
        
        # Final projection -> scalar threshold
        threshold = self.fc(x)  # [B, 1]
        
        return threshold.squeeze(-1)  # [B]


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float
) -> torch.Tensor:
    """
    Pinball loss for quantile regression.
    
    The pinball loss (also called quantile loss) is:
        L_tau(y, y_pred) = max(tau * (y - y_pred), (tau - 1) * (y - y_pred))
    
    This loss is minimized when y_pred is the tau-th quantile of y.
    
    Properties:
    - When y > y_pred: loss = tau * (y - y_pred)
    - When y < y_pred: loss = (tau - 1) * (y - y_pred) = (1 - tau) * (y_pred - y)
    - The loss is asymmetric: over-prediction is penalized differently than under-prediction
    - For tau=0.5, this becomes the absolute error (MAE)
    
    Args:
        pred: Predicted quantile [N] or [B]
        target: Target values [N] or [B]
        tau: Quantile level in [0, 1]
        
    Returns:
        Pinball loss (scalar or per-sample if reduction='none')
        
    Example:
        For tau=0.1 (10th percentile), we want to predict a threshold such that
        10% of scores are below it. The loss penalizes:
        - Under-prediction (pred too low) more heavily: tau * error
        - Over-prediction (pred too high) less: (1-tau) * error
    """
    error = target - pred  # [N] or [B]
    
    # Pinball loss: max(tau * error, (tau - 1) * error)
    loss = torch.max(tau * error, (tau - 1) * error)
    
    return loss

