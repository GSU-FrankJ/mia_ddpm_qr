"""UNet backbone for DDPM/DDIM on CIFAR-10.

The model mirrors the canonical architecture used for 32×32 diffusion models
with sinusoidal timestep embeddings and residual attention blocks at the
requested resolutions. Shapes follow the PyTorch convention `[B, C, H, W]`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


class SinusoidalPosEmb(nn.Module):
    r"""Encode scalar timesteps into sinusoidal embeddings.

    Given integer timesteps `t` (shape `[B]`) we project them into a
    `embedding_dim` vector following the standard diffusion formulation:

    .. math::
        \text{emb}_{2i}(t) = \sin(t / 10^{4i / d})
        \quad\text{and}\quad
        \text{emb}_{2i+1}(t) = \cos(t / 10^{4i / d})

    Args:
        dim: Dimensionality of the output embedding. Must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb expects an even dimension")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            raise ValueError("timesteps must be 1-D tensor [B]")

        device = timesteps.device
        half_dim = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, device=device, dtype=timesteps.dtype)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        # Shape after unsqueeze: [B, half_dim]
        args = timesteps[:, None].float() * frequencies[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with timestep conditioning.

    The block processes feature maps `[B, C, H, W]` with GroupNorm for
    stability and injects the timestep embedding through a learned affine.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_affine = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        time_cond = self.time_affine(time_emb)
        h = h + time_cond[:, :, None, None]
        h = self.conv2(self.dropout(self.activation(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block over spatial grid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        q, k, v = self.qkv(norm_x).chunk(3, dim=1)

        # Flatten spatial dimension; shapes -> [B, H*W, C]
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)

        attention_scores = torch.bmm(q, k) * (c ** -0.5)
        attention_weights = attention_scores.softmax(dim=-1)
        attended = torch.bmm(attention_weights, v)
        attended = attended.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


@dataclass
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    channels: int = 128
    channel_mults: Sequence[int] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: Sequence[int] = (16,)
    dropout: float = 0.0


class UNetModel(nn.Module):
    """Diffusion UNet with residual attention blocks.

    The network consumes noisy images `[B, C, 32, 32]` and integer timesteps
    `[B]`, outputting estimated noise residuals with matching shape.
    """

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.channels),
            nn.Linear(config.channels, config.channels * 4),
            nn.SiLU(),
            nn.Linear(config.channels * 4, config.channels * 4),
        )
        time_dim = config.channels * 4

        self.input_conv = nn.Conv2d(config.in_channels, config.channels, kernel_size=3, padding=1)

        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        self.skip_channel_history: List[int] = []

        in_ch = config.channels
        current_resolution = 32
        for level, mult in enumerate(config.channel_mults):
            block_layers = nn.ModuleList()
            out_ch = config.channels * mult
            for _ in range(config.num_res_blocks):
                block_layers.append(ResidualBlock(in_ch, out_ch, time_dim, config.dropout))
                self.skip_channel_history.append(out_ch)
                if current_resolution in config.attention_resolutions:
                    block_layers.append(AttentionBlock(out_ch))
                in_ch = out_ch
            self.down_blocks.append(block_layers)
            if level != len(config.channel_mults) - 1:
                self.downsamples.append(Downsample(in_ch))
                current_resolution //= 2

        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_dim, config.dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_dim, config.dropout)

        skip_channels = list(reversed(self.skip_channel_history))
        skip_idx = 0
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        for level, mult in reversed(list(enumerate(config.channel_mults))):
            block_layers = nn.ModuleList()
            out_ch = config.channels * mult
            for _ in range(config.num_res_blocks):
                skip_ch = skip_channels[skip_idx]
                skip_idx += 1
                block_layers.append(
                    ResidualBlock(in_ch + skip_ch, out_ch, time_dim, config.dropout)
                )
                if current_resolution in config.attention_resolutions:
                    block_layers.append(AttentionBlock(out_ch))
                in_ch = out_ch
            self.up_blocks.append(block_layers)
            if level != 0:
                self.upsamples.append(Upsample(in_ch))
                current_resolution *= 2

        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, config.out_channels, kernel_size=3, padding=1)

        LOGGER.info(
            "Initialized UNetModel with %s parameters",
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input x must be [B, C, H, W]")
        if timesteps.dim() != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps must be [B] aligned with batch size")

        time_emb = self.time_embed(timesteps)

        skips: List[torch.Tensor] = []
        current = self.input_conv(x)
        for idx, block_layers in enumerate(self.down_blocks):
            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    current = layer(current, time_emb)
                    skips.append(current)
                else:
                    current = layer(current)
            if idx < len(self.downsamples):
                current = self.downsamples[idx](current)

        current = self.mid_block1(current, time_emb)
        current = self.mid_attn(current)
        current = self.mid_block2(current, time_emb)

        for idx, block_layers in enumerate(self.up_blocks):
            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    skip = skips.pop()
                    current = torch.cat([current, skip], dim=1)
                    current = layer(current, time_emb)
                else:
                    current = layer(current)
            if idx < len(self.upsamples):
                current = self.upsamples[idx](current)

        out = self.out_conv(self.out_act(self.out_norm(current)))
        return out


def build_unet(config_dict: Optional[dict] = None) -> UNetModel:
    config = UNetConfig(**(config_dict or {}))
    return UNetModel(config)


__all__ = ["UNetModel", "UNetConfig", "build_unet"]

