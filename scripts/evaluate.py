"""Step 5: per-image metrics for every method on a split.

    python scripts/evaluate.py --split dev
    python scripts/evaluate.py --split test
    python scripts/evaluate.py --split dev --metrics ocr      # recompute one metric

Writes results/metrics/<split>/<method>.csv (and <method>_ocr.json with the
OCR transcriptions). The pseudo-method "hr" scores the HR images themselves.
"""

import argparse

from srbench.config import Paths, load_config
from srbench.data.prepare import load_manifest
from srbench.evaluate import ALL_METRICS, Evaluator, save_metrics
from srbench.models import METHODS
from srbench.runtime import pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--methods", nargs="+", choices=("hr",) + METHODS, default=None)
    ap.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=list(ALL_METRICS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.split)
    paths = Paths(cfg)
    rows = load_manifest(paths, args.split)
    evaluator = Evaluator(cfg, paths, pick_device(args.device), args.metrics)
    for method in args.methods or ["hr"] + cfg["methods"]:
        df, transcripts = evaluator.evaluate(method, rows)
        if df.empty:
            print(f"{method}: no outputs found, skipped")
            continue
        out = paths.metrics(args.split, method)
        save_metrics(df, transcripts, out)
        print(f"{method}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
