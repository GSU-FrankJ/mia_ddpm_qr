"""Quantile regression models for threshold estimation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


class SmallCNNQuantile(nn.Module):
    """Compact CNN that predicts scalar thresholds from images.

    Architecture: three convolutional blocks (Conv-BN-ReLU) expanding from 64
    to 128 channels, followed by global average pooling and a linear head.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `[B, 3, 32, 32]` inputs to `[B]` quantile predictions."""

        features = self.features(x)
        return self.head(features).squeeze(-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Quantile (pinball) loss encouraging the τ-quantile estimate.

    The loss is asymmetric: residuals above the predicted quantile incur a
    scaled positive penalty, while residuals below incur a complementary
    penalty. This drives the network to output the desired quantile of the
    target distribution.
    """

    residual = target - pred
    loss = torch.maximum(tau * residual, (tau - 1) * residual)
    return loss.mean()

