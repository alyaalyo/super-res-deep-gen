"""Cost: seconds per image and peak accelerator memory, per method.

    python scripts/benchmark_runtime.py                 # current machine
    python scripts/benchmark_runtime.py --device cuda   # on the Colab/Kaggle T4

Upscales N 128x128 dev inputs (after warm-up) one at a time, batch size 1, and
writes results/runtime/<device>.csv. Model loading and disk I/O are not timed.
"""

import argparse
import re

import pandas as pd

from srbench.config import Paths, load_config
from srbench.data.prepare import load_manifest
from srbench.imageio import imread
from srbench.models import METHODS, build_model
from srbench.runtime import Meter, device_name, pick_device
from srbench.stats import bootstrap_ci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config("dev")
    paths = Paths(cfg)
    rows = load_manifest(paths, "dev").sample(frac=1, random_state=0).head(args.n)
    lrs = [imread(paths.lr("none", r.content, r.id)) for r in rows.itertuples()]
    device = pick_device(args.device)
    dev = device_name(device)

    records = []
    for name in args.methods or cfg["methods"]:
        model = build_model(name, cfg, device)
        for lr in lrs[: args.warmup]:
            model(lr, seed=0)
        meter = Meter(device)
        for i, lr in enumerate(lrs):
            with meter.measure():
                model(lr, seed=i)
            records.append(dict(method=name, device=dev, seconds=meter.seconds, peak_mb=meter.peak_mb))
        df = pd.DataFrame([r for r in records if r["method"] == name])
        m, lo, hi = bootstrap_ci(df.seconds)
        print(f"{name:12s} {m:.3f} s/img [{lo:.3f}, {hi:.3f}]  peak {df.peak_mb.max():.0f} MB  ({dev})")
        del model

    out = paths.results / "runtime" / f"{re.sub(r'[^A-Za-z0-9]+', '_', dev).strip('_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
