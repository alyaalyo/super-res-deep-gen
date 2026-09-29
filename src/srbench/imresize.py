"""MATLAB-compatible bicubic ``imresize`` (antialiased when downscaling).

SR benchmarks create LR images with MATLAB's ``imresize(img, 1/4, 'bicubic')``.
PIL and OpenCV use slightly different kernels/antialiasing, which shifts PSNR,
so we port MATLAB's algorithm: cubic kernel with a = -0.5, kernel stretched by
1/scale when downscaling, symmetric boundary handling, and the dimension with
the smaller scale resized first.
"""

from __future__ import annotations

import math

import numpy as np

from .imageio import to_uint8


def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    ax2, ax3 = ax**2, ax**3
    return (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))


def _contributions(in_len: int, out_len: int, scale: float, antialias: bool = True):
    kernel_width = 4.0
    if scale < 1 and antialias:
        def kernel(x):
            return scale * _cubic(scale * x)
        kernel_width /= scale
    else:
        kernel = _cubic

    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)          # output pixel centres in input coordinates
    left = np.floor(u - kernel_width / 2)
    p = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(p)[None, :]  # 1-based, like MATLAB
    weights = kernel(u[:, None] - indices)
    weights = weights / weights.sum(axis=1, keepdims=True)

    keep = np.any(weights != 0, axis=0)
    weights, indices = weights[:, keep], indices[:, keep]

    # symmetric padding: mirror out-of-range indices back into the image
    aux = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    indices = aux[np.mod(indices.astype(np.int64) - 1, aux.size)]
    return weights, indices


def _resize_axis(img: np.ndarray, axis: int, out_len: int, scale: float) -> np.ndarray:
    weights, indices = _contributions(img.shape[axis], out_len, scale)
    moved = np.moveaxis(img, axis, 0)
    gathered = moved[indices]                        # out_len x P x ...
    w = weights.reshape(weights.shape + (1,) * (moved.ndim - 1))
    return np.moveaxis((gathered * w).sum(axis=1), 0, axis)


def imresize(img: np.ndarray, scale: float | None = None, output_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Resize an HxW[xC] image. Returns float64 in the input's value range."""
    if (scale is None) == (output_shape is None):
        raise ValueError("pass exactly one of scale or output_shape")
    h, w = img.shape[:2]
    if output_shape is None:
        scales = (scale, scale)
        output_shape = (int(math.ceil(h * scale)), int(math.ceil(w * scale)))
    else:
        scales = (output_shape[0] / h, output_shape[1] / w)

    out = img.astype(np.float64)
    for axis in np.argsort(scales, kind="stable"):
        out = _resize_axis(out, int(axis), output_shape[axis], scales[axis])
    return out


def imresize_uint8(img: np.ndarray, scale: float | None = None, output_shape: tuple[int, int] | None = None) -> np.ndarray:
    return to_uint8(imresize(img, scale=scale, output_shape=output_shape))
