"""Image I/O. Images are always HxWx3 uint8 RGB numpy arrays."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def imread(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def imwrite(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="PNG")


def jpeg_roundtrip(img: np.ndarray, quality: int, subsampling: int = 2) -> np.ndarray:
    """Encode with baseline libjpeg at ``quality`` and decode again."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Round a float image in [0, 255] to uint8 (MATLAB-style rounding)."""
    return np.clip(np.round(img), 0, 255).astype(np.uint8)
