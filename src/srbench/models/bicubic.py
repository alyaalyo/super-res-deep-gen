from __future__ import annotations

import numpy as np

from ..imresize import imresize_uint8
from .base import SRModel


class Bicubic(SRModel):
    """MATLAB-style bicubic interpolation (the same kernel used to make the LR)."""

    name = "bicubic"

    def upscale(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        return imresize_uint8(lr, scale=self.scale)
