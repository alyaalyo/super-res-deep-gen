"""Step 1: download DIV2K validation, the FFHQ subset and SR model weights.

    python scripts/download.py [--skip-data] [--skip-weights]
"""

import argparse

from srbench.config import Paths, load_config
from srbench.data import download
from srbench.data.prepare import face_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    paths = Paths(cfg)
    if not args.skip_data:
        o = cfg["data"]["objects"]
        print("DIV2K ->", download.download_div2k_valid(paths.raw, o["first_id"], o["last_id"]))
        ids = sorted(sum(face_split(cfg).values(), []))
        print("FFHQ ->", download.download_ffhq(paths.raw, ids))
    if not args.skip_weights:
        for name, path in download.download_weights(cfg, paths.weights).items():
            print(f"{name:15s} -> {path}")


if __name__ == "__main__":
    main()
