from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18QR(nn.Module):
    def __init__(self, num_outputs: int, stats_dim: int = 4, dropout: float = 0.0):
        super().__init__()
        if stats_dim <= 0:
            raise ValueError("stats_dim must be positive")
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        hidden_features = in_features // 2
        layers = [
            nn.Linear(in_features + stats_dim, in_features),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_features, num_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        combined = torch.cat([features, stats], dim=1)
        return self.head(combined)
