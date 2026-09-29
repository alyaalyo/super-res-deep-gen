"""LPIPS (full-reference, lower is better) and MUSIQ (no-reference, higher is better)."""

from __future__ import annotations

import numpy as np
import torch

from ..models.base import to_tensor


class LPIPS:
    def __init__(self, device: torch.device, net: str = "alex"):
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.inference_mode()
    def __call__(self, img: np.ndarray, ref: np.ndarray) -> float:
        a = to_tensor(img, self.device) * 2 - 1
        b = to_tensor(ref, self.device) * 2 - 1
        return float(self.model(a, b).item())


class MUSIQ:
    """MUSIQ trained on KonIQ-10k (pyiqa default), score range ~0-100."""

    def __init__(self, device: torch.device):
        import pyiqa

        self.device = device
        self.model = pyiqa.create_metric("musiq", device=device, as_loss=False)

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> float:
        return float(self.model(to_tensor(img, self.device)).item())
