"""Running SR methods over the dataset, with the test-split run-once guard."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from tqdm import tqdm

from .config import Paths, config_hash, jpeg_tag
from .imageio import imread, imwrite
from .models import SRModel, image_seed
from .runtime import Meter, device_name

RUNTIME_FIELDS = ["id", "content", "jpeg", "method", "device", "seconds", "peak_mb"]


def run_method(model: SRModel, rows: pd.DataFrame, jpegs: list[int | None], paths: Paths,
               out_path: Callable[[str, str, str], Path], seed_base: int = 0,
               runtime_log: Path | None = None, force: bool = False) -> int:
    """Upscale every (image, JPEG level); skips existing outputs so runs can resume.

    ``out_path(jpeg_tag, content, id)`` gives the destination. Returns the
    number of images processed.
    """
    jobs = [(r, jpeg_tag(q)) for r in rows.itertuples() for q in jpegs]
    todo = [(r, t) for r, t in jobs if force or not out_path(t, r.content, r.id).exists()]
    if not todo:
        return 0

    # warm-up (kernel selection, lazy init) so it is not counted in the timings
    model(imread(paths.lr(todo[0][1], todo[0][0].content, todo[0][0].id)), seed=0)

    log_file = writer = None
    if runtime_log is not None:
        runtime_log.parent.mkdir(parents=True, exist_ok=True)
        new = not runtime_log.exists()
        log_file = open(runtime_log, "a", newline="")
        writer = csv.DictWriter(log_file, RUNTIME_FIELDS)
        if new:
            writer.writeheader()

    meter, dev = Meter(model.device), device_name(model.device)
    try:
        for r, tag in tqdm(todo, desc=model.name):
            lr = imread(paths.lr(tag, r.content, r.id))
            # same noise for all JPEG levels of an image: the JPEG sweep is paired
            seed = image_seed(seed_base, r.id)
            with meter.measure():
                sr = model(lr, seed=seed)
            imwrite(out_path(tag, r.content, r.id), sr)
            if writer:
                writer.writerow(dict(id=r.id, content=r.content, jpeg=tag, method=model.name, device=dev,
                                     seconds=round(meter.seconds, 5), peak_mb=round(meter.peak_mb, 1)))
                log_file.flush()
    finally:
        if log_file:
            log_file.close()
        if model.device.type == "mps":
            torch.mps.empty_cache()
    return len(todo)


class TestLock:
    """Records which methods have been run on the test split and with which
    settings, so the test split is run once, after settings are frozen."""

    __test__ = False   # not a pytest test class

    def __init__(self, paths: Paths, cfg: dict):
        self.path = paths.test_lock
        self.hash = config_hash(cfg)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def check(self, method: str, force: bool) -> bool:
        """True if the method should run now."""
        prev = self.state.get(method)
        if prev is None:
            return True
        if prev["config_hash"] != self.hash and not force:
            raise SystemExit(
                f"[test lock] {method} was already run on test with config {prev['config_hash']}; the "
                f"current config is {self.hash}. The test split is run once. Use --force only to fix a bug, "
                "and say so in the report."
            )
        if not prev.get("finished"):
            return True          # resuming an interrupted run with the same settings
        if force:
            print(f"[test lock] --force: re-running {method} on test (previous run {prev['finished']})")
            return True
        print(f"[test lock] {method} already done on test ({prev['finished']}), skipping")
        return False

    def mark(self, method: str, stage: str, device: str) -> None:
        entry = self.state.setdefault(method, {})
        entry.update(config_hash=self.hash, device=device)
        entry[stage] = time.strftime("%Y-%m-%d %H:%M:%S")
        if stage == "started":
            entry.pop("finished", None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))
