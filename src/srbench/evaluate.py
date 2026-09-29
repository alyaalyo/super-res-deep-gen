"""Per-image metrics for one method on one split.

Every method is scored against the HR image. The pseudo-method ``hr`` scores
the HR image itself; it gives the ceilings for the no-reference and task
metrics (MUSIQ of real photos, OCR error on clean text).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Paths, jpeg_levels, jpeg_tag
from .imageio import imread
from .metrics.fidelity import psnr_y, ssim_y
from .metrics.ocr import OCRConfig, cer, read_text, wer

ALL_METRICS = ("psnr", "ssim", "lpips", "musiq", "ocr", "face")
KEYS = ["id", "content", "jpeg", "method"]


def ocr_config(cfg: dict) -> OCRConfig:
    o = cfg["metrics"]["ocr"]
    return OCRConfig(psm=int(o["psm"]), upscale=int(o["upscale"]), dictionary=bool(o["dictionary"]))


class FaceCache:
    """HR landmarks (cached on disk) and HR embeddings for the ArcFace metric."""

    def __init__(self, paths: Paths):
        from .metrics.face import ArcFace

        self.paths = paths
        self.model = ArcFace(paths.weights / "insightface")
        self.kps_file = paths.results / "cache" / "face_kps.json"
        self.kps = json.loads(self.kps_file.read_text()) if self.kps_file.exists() else {}
        self._emb: dict[str, np.ndarray] = {}

    def landmarks(self, image_id: str) -> np.ndarray:
        if image_id not in self.kps:
            kps, source = self.model.landmarks(imread(self.paths.hr("faces", image_id)))
            if source != "detector@0.5":
                print(f"[face] {image_id}: landmarks from {source}")
            self.kps[image_id] = {"kps": kps.tolist(), "source": source}
            self.kps_file.parent.mkdir(parents=True, exist_ok=True)
            self.kps_file.write_text(json.dumps(self.kps, indent=1))
        return np.asarray(self.kps[image_id]["kps"], dtype=np.float32)

    def embed(self, img: np.ndarray, image_id: str) -> np.ndarray:
        return self.model.embed(img, self.landmarks(image_id))

    def hr_embedding(self, image_id: str) -> np.ndarray:
        if image_id not in self._emb:
            self._emb[image_id] = self.embed(imread(self.paths.hr("faces", image_id)), image_id)
        return self._emb[image_id]

    def similarity(self, img: np.ndarray, image_id: str) -> float:
        return float(self.embed(img, image_id) @ self.hr_embedding(image_id))


class Evaluator:
    def __init__(self, cfg: dict, paths: Paths, device: torch.device, metrics=ALL_METRICS):
        self.cfg, self.paths, self.device = cfg, paths, device
        self.metrics = set(metrics)
        self.crop = cfg["metrics"]["crop_border"]
        self.lpips = self.musiq = self.faces = None
        if "lpips" in self.metrics:
            from .metrics.perceptual import LPIPS
            self.lpips = LPIPS(device, cfg["metrics"]["lpips_net"])
        if "musiq" in self.metrics:
            from .metrics.perceptual import MUSIQ
            self.musiq = MUSIQ(device)
        if "face" in self.metrics:
            self.faces = FaceCache(paths)
        self.text_gt = json.loads(paths.text_gt.read_text()) if paths.text_gt.exists() else {}
        self.ocr_cfg = ocr_config(cfg)

    def image_path(self, method: str, tag: str, content: str, image_id: str) -> Path:
        return self.paths.hr(content, image_id) if method == "hr" else self.paths.sr(method, tag, content, image_id)

    def evaluate(self, method: str, rows: pd.DataFrame, jpegs: list[int | None] | None = None,
                 image_path=None) -> tuple[pd.DataFrame, dict]:
        """Returns (per-image metrics, {"id/jpeg": OCR transcription})."""
        image_path = image_path or self.image_path
        tags = ["none"] if method == "hr" else [jpeg_tag(q) for q in (jpegs or jpeg_levels(self.cfg))]
        records, ocr_jobs = [], []
        for r in tqdm(list(rows.itertuples()), desc=f"eval {method}"):
            hr = imread(self.paths.hr(r.content, r.id))
            for tag in tags:
                path = image_path(method, tag, r.content, r.id)
                if not path.exists():
                    continue
                img = imread(path)
                rec = dict(id=r.id, content=r.content, jpeg=tag, method=method)
                if "psnr" in self.metrics:
                    rec["psnr"] = psnr_y(img, hr, self.crop)
                if "ssim" in self.metrics:
                    rec["ssim"] = ssim_y(img, hr, self.crop)
                if self.lpips:
                    rec["lpips"] = self.lpips(img, hr)
                if self.musiq:
                    rec["musiq"] = self.musiq(img)
                if self.faces and r.content == "faces":
                    rec["face_sim"] = self.faces.similarity(img, r.id)
                if "ocr" in self.metrics and r.content == "text":
                    ocr_jobs.append((len(records), path, r.id))
                records.append(rec)

        transcripts = {}
        if ocr_jobs:
            texts = self._ocr([p for _, p, _ in ocr_jobs])
            for (i, _, image_id), text in zip(ocr_jobs, texts):
                gt = self.text_gt[image_id]
                records[i]["cer"] = cer(text, gt)
                records[i]["wer"] = wer(text, gt)
                transcripts[f"{image_id}/{records[i]['jpeg']}"] = text
        return pd.DataFrame(records), transcripts

    def _ocr(self, image_paths: list[Path]) -> list[str]:
        workers = max(1, (os.cpu_count() or 2) - 1)
        with ThreadPoolExecutor(workers) as pool:
            return list(tqdm(pool.map(lambda p: read_text(imread(p), self.ocr_cfg), image_paths),
                             total=len(image_paths), desc="OCR", leave=False))


def save_metrics(df: pd.DataFrame, transcripts: dict, out: Path) -> pd.DataFrame:
    """Write per-image metrics; columns not recomputed this time are kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        keep = [c for c in old.columns if c not in df.columns or c in KEYS]
        df = old[keep].merge(df, on=KEYS, how="outer")
    df.sort_values(KEYS).to_csv(out, index=False)
    if transcripts:
        tfile = out.with_name(out.stem + "_ocr.json")
        old = json.loads(tfile.read_text()) if tfile.exists() else {}
        tfile.write_text(json.dumps({**old, **transcripts}, indent=1))
    return df


def load_metrics(paths: Paths, split: str) -> pd.DataFrame:
    files = sorted((paths.results / "metrics" / split).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no metrics for {split}; run scripts/evaluate.py --split {split}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
