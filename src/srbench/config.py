"""Configuration loading, paths and the frozen-config guard for the test split."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
FROZEN_CONFIG = ROOT / "configs" / "frozen.yaml"

CONTENTS = ("objects", "faces", "text")
SPLITS = ("dev", "test")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(split: str | None = None, path: str | Path | None = None) -> dict:
    """Load the experiment config.

    The test split always uses ``configs/frozen.yaml`` (written by
    ``scripts/tune_dev.py``) so that no setting can change after test outputs
    have been looked at. The dev split uses the frozen config if it exists,
    otherwise the defaults.
    """
    if path is not None:
        return load_yaml(Path(path))
    if split == "test":
        if not FROZEN_CONFIG.exists():
            raise SystemExit(
                "configs/frozen.yaml not found. Tune settings on the dev split first "
                "(scripts/tune_dev.py); the test split only runs with frozen settings."
            )
        return load_yaml(FROZEN_CONFIG)
    return load_yaml(FROZEN_CONFIG if FROZEN_CONFIG.exists() else DEFAULT_CONFIG)


def config_hash(cfg: dict) -> str:
    """Stable hash of the settings, ignoring provenance metadata."""
    c = copy.deepcopy(cfg)
    c.pop("frozen", None)
    blob = json.dumps(c, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def jpeg_levels(cfg: dict) -> list[int | None]:
    return [None if str(q).lower() in ("none", "null") else int(q) for q in cfg["data"]["jpeg_qualities"]]


def jpeg_tag(q: int | None) -> str:
    return "none" if q is None else f"q{q}"


def parse_jpeg_tag(tag: str) -> int | None:
    return None if tag == "none" else int(tag.lstrip("q"))


class Paths:
    """All on-disk locations used by the pipeline."""

    def __init__(self, cfg: dict):
        p = cfg["paths"]
        self.data = ROOT / p["data"]
        self.weights = ROOT / p["weights"]
        self.outputs = ROOT / p["outputs"]
        self.results = ROOT / p["results"]
        self.raw = self.data / "raw"
        self.manifest = self.data / "manifest.csv"
        self.text_gt = self.data / "text_gt.json"
        self.test_lock = self.outputs / "test_runs.json"

    def hr(self, content: str, image_id: str) -> Path:
        return self.data / "hr" / content / f"{image_id}.png"

    def lr(self, jpeg: str, content: str, image_id: str) -> Path:
        return self.data / "lr" / jpeg / content / f"{image_id}.png"

    def sr(self, method: str, jpeg: str, content: str, image_id: str) -> Path:
        return self.outputs / "sr" / method / jpeg / content / f"{image_id}.png"

    def runtime_log(self, split: str, method: str) -> Path:
        return self.outputs / "runtime" / f"{split}_{method}.csv"

    def metrics(self, split: str, method: str) -> Path:
        return self.results / "metrics" / split / f"{method}.csv"
