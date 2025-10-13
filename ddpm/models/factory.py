from __future__ import annotations

from typing import Any, Dict

from ddpm.models.unet import UNetModel

ARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    "unet_small": {"base_channels": 128, "channel_mults": [1, 2, 2, 4], "num_res_blocks": 2, "dropout": 0.0},
}


def build_unet(arch: str, img_size: int, overrides: Dict[str, Any] | None = None) -> UNetModel:
    params = {**ARCH_PRESETS.get(arch, {}), **(overrides or {})}
    return UNetModel(
        img_size=img_size,
        in_channels=params.get("in_channels", 3),
        out_channels=params.get("out_channels", 3),
        base_channels=params.get("base_channels", 128),
        channel_mults=params.get("channel_mults", (1, 2, 2, 4)),
        num_res_blocks=params.get("num_res_blocks", 2),
        dropout=params.get("dropout", 0.0),
        attn_resolutions=params.get("attn_resolutions", (16,)),
    )

