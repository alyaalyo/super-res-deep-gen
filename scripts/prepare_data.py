"""Step 2: HR crops, synthetic text, dev/test split and degraded LR inputs.

    python scripts/prepare_data.py

Writes data/hr/, data/lr/{none,q50,q30,q10}/, data/manifest.csv and
data/text_gt.json.
"""

import argparse

from srbench.config import Paths, load_config
from srbench.data.prepare import build_hr, build_lr, check_leakage


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    paths = Paths(cfg)
    manifest = build_hr(cfg, paths)
    check_leakage(manifest)
    build_lr(cfg, paths, manifest)
    print(f"manifest: {paths.manifest} ({len(manifest)} images)")


if __name__ == "__main__":
    main()
