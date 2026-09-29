"""Tesseract OCR and character / word error rates against the exact transcription."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image
from rapidfuzz.distance import Levenshtein

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # we parallelise across images instead


@dataclass(frozen=True)
class OCRConfig:
    psm: int = 6              # 6 = a single uniform block of text
    upscale: int = 2          # resize before OCR (Tesseract prefers ~20-30 px x-height)
    dictionary: bool = True   # Tesseract's word lists (language prior)

    def tesseract_args(self) -> str:
        args = f"--oem 1 --psm {self.psm}"
        if not self.dictionary:
            args += " -c load_system_dawg=0 -c load_freq_dawg=0"
        return args


def normalize(text: str) -> str:
    """Line breaks and repeated spaces are not errors: collapse all whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def cer(pred: str, gt: str) -> float:
    """Levenshtein distance / reference length. Can exceed 1 for hallucinated text."""
    p, g = normalize(pred), normalize(gt)
    return Levenshtein.distance(p, g) / max(len(g), 1)


def wer(pred: str, gt: str) -> float:
    p, g = normalize(pred).split(), normalize(gt).split()
    return Levenshtein.distance(p, g) / max(len(g), 1)


def read_text(img: np.ndarray, cfg: OCRConfig) -> str:
    im = Image.fromarray(img).convert("L")
    if cfg.upscale != 1:
        im = im.resize((im.width * cfg.upscale, im.height * cfg.upscale), Image.BICUBIC)
    return pytesseract.image_to_string(im, lang="eng", config=cfg.tesseract_args())


def read_many(images: list[np.ndarray], cfg: OCRConfig, workers: int | None = None) -> list[str]:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(workers) as pool:   # pytesseract runs a subprocess per call
        return list(pool.map(lambda im: read_text(im, cfg), images))
