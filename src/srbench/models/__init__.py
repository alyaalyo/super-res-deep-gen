"""Model registry: ``build_model(name, cfg, device)``."""

from __future__ import annotations

import torch

from ..config import Paths
from .base import SRModel, image_seed

METHODS = ("bicubic", "realesrnet", "realesrgan", "resshift", "sd_upscaler")
LABELS = {
    "hr": "HR (reference)",
    "bicubic": "Bicubic",
    "realesrnet": "Real-ESRNet",
    "realesrgan": "Real-ESRGAN",
    "resshift": "ResShift",
    "sd_upscaler": "SD x4 Upscaler",
}


def build_model(name: str, cfg: dict, device: torch.device) -> SRModel:
    weights = Paths(cfg).weights
    opts = cfg["models"].get(name, {})
    if name == "bicubic":
        from .bicubic import Bicubic
        return Bicubic(device)
    if name == "realesrnet":
        from .rrdbnet import RealESRNet
        return RealESRNet(device, weights, opts.get("half_on_cuda", True))
    if name == "realesrgan":
        from .rrdbnet import RealESRGAN
        return RealESRGAN(device, weights, opts.get("half_on_cuda", True))
    if name == "resshift":
        from .resshift import ResShift
        return ResShift(device, weights, **{k: v for k, v in opts.items() if not k.endswith("url")})
    if name == "sd_upscaler":
        from .sd_upscaler import SDUpscaler
        return SDUpscaler(device, **opts)
    raise ValueError(f"unknown method {name!r}; choose from {METHODS}")


__all__ = ["METHODS", "LABELS", "SRModel", "build_model", "image_seed"]
