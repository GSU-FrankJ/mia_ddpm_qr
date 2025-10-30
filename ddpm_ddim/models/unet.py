"""
UNet architecture for DDIM on CIFAR-10 (32x32).

Implements a standard UNet with:
- Downsampling blocks (encoder)
- Upsampling blocks (decoder)
- Skip connections
- Attention at specific resolutions
- Time embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.
    
    Maps timestep t to a d-dimensional embedding using sinusoidal functions.
    This allows the model to condition on the diffusion timestep.
    
    Args:
        dim: Embedding dimension (default: 512)
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding.
        
        Args:
            time: Timestep tensor [B] with values in [0, T-1]
            
        Returns:
            Time embedding [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        # Create exponential decay sequence: [0, 1, ..., half_dim-1]
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        # [B, half_dim] = [B, 1] * [half_dim]
        emb = time[:, None].float() * emb[None, :]
        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """
    Residual block with time conditioning.
    
    Architecture:
    - GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv -> Residual
    
    Args:
        channels: Number of input/output channels
        time_emb_dim: Time embedding dimension
        dropout: Dropout probability
        use_scale_shift_norm: If True, apply scale and shift from time embedding
    """
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Time embedding projection
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2 if use_scale_shift_norm else channels)
        )
        
        # Main conv blocks
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map [B, C, H, W]
            time_emb: Time embedding [B, time_emb_dim]
            
        Returns:
            Output feature map [B, C, H, W]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Condition on time embedding
        time_emb = self.time_emb(time_emb)
        if self.use_scale_shift_norm:
            # Apply scale and shift
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class AttentionBlock(nn.Module):
    """
    Self-attention block for spatial features.
    
    Implements multi-head self-attention over spatial dimensions.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Output feature map [B, C, H, W]
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention: [B, C, H*W] -> [B, H*W, C]
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)
        
        # Attention: Q @ K^T / sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.bmm(attn, v)
        h = h.transpose(1, 2).view(B, C, H, W)
        
        # Project and residual
        h = self.proj(h)
        return x + h


class DownBlock(nn.Module):
    """
    Downsampling block for encoder.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Time embedding dimension
        dropout: Dropout probability
        use_attention: Whether to use attention
        use_scale_shift_norm: Whether to use scale-shift norm
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_attention: bool = False,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(in_channels, time_emb_dim, dropout, use_scale_shift_norm),
            ResBlock(out_channels, time_emb_dim, dropout, use_scale_shift_norm)
        ])
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map [B, C_in, H, W]
            time_emb: Time embedding [B, time_emb_dim]
            
        Returns:
            Output feature map [B, C_out, H//2, W//2]
        """
        h = self.downsample(x)
        h = self.resblocks[0](h, time_emb)
        h = self.resblocks[1](h, time_emb)
        h = self.attention(h)
        return h


class UpBlock(nn.Module):
    """
    Upsampling block for decoder.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Time embedding dimension
        dropout: Dropout probability
        use_attention: Whether to use attention
        use_scale_shift_norm: Whether to use scale-shift norm
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_attention: bool = False,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(in_channels, time_emb_dim, dropout, use_scale_shift_norm),
            ResBlock(out_channels, time_emb_dim, dropout, use_scale_shift_norm)
        ])
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map [B, C_in, H, W]
            time_emb: Time embedding [B, time_emb_dim]
            
        Returns:
            Output feature map [B, C_out, H*2, W*2]
        """
        h = self.upsample(x)
        h = self.resblocks[0](h, time_emb)
        h = self.resblocks[1](h, time_emb)
        h = self.attention(h)
        return h


class UNet(nn.Module):
    """
    UNet model for DDIM on CIFAR-10.
    
    Architecture:
    - Encoder: Downsampling blocks with attention at specific resolutions
    - Bottleneck: ResBlocks
    - Decoder: Upsampling blocks with skip connections
    
    Args:
        in_channels: Input image channels (3 for RGB)
        out_channels: Output channels (3 for RGB)
        model_channels: Base channel count
        channel_mult: Channel multipliers per resolution level
        num_res_blocks: Number of ResBlocks per level
        attention_resolutions: Resolutions to apply attention (e.g., [16] for 16x16)
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use scale-shift norm in ResBlocks
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        resolution = 32  # CIFAR-10 is 32x32
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for i in range(num_res_blocks):
                # Check if we should use attention at this resolution
                use_attn = resolution in attention_resolutions
                self.down_blocks.append(
                    ResBlock(ch, time_emb_dim, dropout, use_scale_shift_norm)
                )
                ch = out_ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:  # Don't downsample after last level
                self.down_blocks.append(
                    DownBlock(ch, ch, time_emb_dim, dropout, use_attn, use_scale_shift_norm)
                )
                input_block_chans.append(ch)
                resolution //= 2
        
        # Bottleneck
        self.middle_block1 = ResBlock(ch, time_emb_dim, dropout, use_scale_shift_norm)
        self.middle_attn = AttentionBlock(ch) if 16 in attention_resolutions else nn.Identity()
        self.middle_block2 = ResBlock(ch, time_emb_dim, dropout, use_scale_shift_norm)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                use_attn = resolution in attention_resolutions
                ich = input_block_chans.pop()
                if i == 0:
                    # First block in level: upsample
                    self.up_blocks.append(
                        UpBlock(ch, out_ch, time_emb_dim, dropout, use_attn, use_scale_shift_norm)
                    )
                    ch = out_ch
                else:
                    # Subsequent blocks: residual
                    self.up_blocks.append(
                        ResBlock(ch + ich, time_emb_dim, dropout, use_scale_shift_norm)
                    )
                    ch = out_ch
                if level != len(channel_mult) - 1:
                    resolution *= 2
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy image [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.input_conv(x)
        hs = [h]
        
        # Encoder
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, time_emb)
                hs.append(h)
            elif isinstance(block, DownBlock):
                h = block(h, time_emb)
                hs.append(h)
        
        # Bottleneck
        h = self.middle_block1(h, time_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, time_emb)
        
        # Decoder
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                # Concatenate skip connection
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, time_emb)
            elif isinstance(block, UpBlock):
                h = block(h, time_emb)
                h = torch.cat([h, hs.pop()], dim=1)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h

