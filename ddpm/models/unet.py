from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        timesteps = timesteps.float()
        device = timesteps.device
        emb = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -emb)
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        time_term = self.time_emb(t_emb)[:, :, None, None]
        h = h + time_term
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = q.permute(0, 1, 3, 2)  # b, heads, hw, dim
        k = k.permute(0, 1, 2, 3)  # b, heads, dim, hw
        attn = torch.matmul(q, k) / math.sqrt(c // self.num_heads)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = conv3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownStage(nn.Module):
    def __init__(
        self,
        blocks: Sequence[ResidualBlock],
        attn: Optional[nn.Module],
        downsample: Optional[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attn = attn if attn is not None else nn.Identity()
        self.downsample = downsample

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skip_stack: list[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, t_emb)
            skip_stack.append(x)
        x = self.attn(x)
        if skip_stack:
            skip_stack[-1] = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpStage(nn.Module):
    def __init__(
        self,
        blocks: Sequence[ResidualBlock],
        attn: Optional[nn.Module],
        upsample: Optional[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attn = attn if attn is not None else nn.Identity()
        self.upsample = upsample

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skip_stack: list[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            skip = skip_stack.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        attn_resolutions: Iterable[int] = (16,),
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)

        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = conv3x3(in_channels, base_channels)

        # Down path
        self.down_stages = nn.ModuleList()
        skip_channels: list[int] = []
        curr_channels = base_channels
        resolution = img_size
        for i, mult in enumerate(channel_mults):
            out_channels_stage = base_channels * mult
            blocks: list[ResidualBlock] = []
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(curr_channels, out_channels_stage, time_dim, dropout))
                curr_channels = out_channels_stage
                skip_channels.append(curr_channels)
            attn = AttentionBlock(curr_channels) if resolution in self.attn_resolutions else nn.Identity()
            downsample = Downsample(curr_channels) if i != len(channel_mults) - 1 else None
            self.down_stages.append(DownStage(blocks, attn, downsample))
            if downsample is not None:
                resolution //= 2

        # Middle
        self.mid_block1 = ResidualBlock(curr_channels, curr_channels, time_dim, dropout)
        self.mid_attn = AttentionBlock(curr_channels)
        self.mid_block2 = ResidualBlock(curr_channels, curr_channels, time_dim, dropout)

        # Up path
        self.up_stages = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels_stage = base_channels * mult
            blocks: list[ResidualBlock] = []
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                blocks.append(ResidualBlock(curr_channels + skip_ch, out_channels_stage, time_dim, dropout))
                curr_channels = out_channels_stage
            attn = AttentionBlock(curr_channels) if (resolution in self.attn_resolutions) else nn.Identity()
            upsample = Upsample(curr_channels) if i != 0 else None
            if upsample is not None:
                resolution *= 2
            self.up_stages.append(UpStage(blocks, attn, upsample))

        self.out_norm = nn.GroupNorm(32, curr_channels)
        self.out_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        h = self.init_conv(x)
        skip_stack: list[torch.Tensor] = []

        for stage in self.down_stages:
            h = stage(h, t_emb, skip_stack)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for stage in self.up_stages:
            h = stage(h, t_emb, skip_stack)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)

