"""Common interface for all x4 SR methods."""

from __future__ import annotations

import zlib

import numpy as np
import torch


class SRModel:
    """``upscale`` maps an HxWx3 uint8 RGB image to a 4Hx4Wx3 uint8 RGB image."""

    name: str = "base"
    scale: int = 4
    stochastic: bool = False

    def __init__(self, device: torch.device):
        self.device = device

    def upscale(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        out = self.upscale(lr, seed=seed)
        expected = (lr.shape[0] * self.scale, lr.shape[1] * self.scale, 3)
        assert out.shape == expected and out.dtype == np.uint8, (out.shape, out.dtype)
        return out


def image_seed(base: int, image_key: str) -> int:
    """Per-image seed that does not depend on processing order."""
    return (base * 1_000_003 + zlib.crc32(image_key.encode())) % 2**31


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(255).to(device)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """1x3xHxW tensor in [0, 1] -> HxWx3 uint8."""
    arr = t[0].float().clamp_(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)
