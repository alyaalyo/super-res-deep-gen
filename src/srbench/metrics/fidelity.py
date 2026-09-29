"""PSNR and SSIM on the Y channel with a border crop (standard SR protocol,
matching BasicSR / the MATLAB evaluation scripts)."""

from __future__ import annotations

import cv2
import numpy as np


def rgb2y(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> BT.601 luma in [16, 235] as float64 (MATLAB ``rgb2ycbcr``)."""
    x = img.astype(np.float64) / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def _crop(y: np.ndarray, border: int) -> np.ndarray:
    return y[border:-border, border:-border] if border > 0 else y


def psnr_y(sr: np.ndarray, hr: np.ndarray, crop_border: int = 4) -> float:
    a, b = _crop(rgb2y(sr), crop_border), _crop(rgb2y(hr), crop_border)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.T)
    mu1 = cv2.filter2D(a, -1, w)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, w)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    s1 = cv2.filter2D(a**2, -1, w)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(b**2, -1, w)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(a * b, -1, w)[5:-5, 5:-5] - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(ssim_map.mean())


def ssim_y(sr: np.ndarray, hr: np.ndarray, crop_border: int = 4) -> float:
    return _ssim(_crop(rgb2y(sr), crop_border), _crop(rgb2y(hr), crop_border))
