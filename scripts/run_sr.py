"""Step 3: run the SR methods on one split (every image x every JPEG level).

    python scripts/run_sr.py --split dev
    python scripts/run_sr.py --split test            # needs configs/frozen.yaml

Outputs go to outputs/sr/<method>/<jpeg>/<content>/<id>.png, per-image
runtimes to outputs/runtime/<split>_<method>.csv. Interrupted runs resume.
"""

import argparse

from srbench.config import Paths, jpeg_levels, load_config
from srbench.data.prepare import load_manifest
from srbench.models import METHODS, build_model
from srbench.pipeline import TestLock, run_method
from srbench.runtime import device_name, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=None, help="default: methods in config")
    ap.add_argument("--contents", nargs="+", choices=["objects", "faces", "text"], default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (debugging, dev only)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    cfg = load_config(args.split)
    paths = Paths(cfg)
    rows = load_manifest(paths, args.split, args.contents)
    if args.limit:
        if args.split == "test":
            raise SystemExit("--limit is for debugging on dev only")
        rows = rows.groupby("content").head(args.limit)
    device = pick_device(args.device)
    lock = TestLock(paths, cfg) if args.split == "test" else None
    print(f"{args.split}: {len(rows)} images x {len(jpeg_levels(cfg))} JPEG levels on {device_name(device)}")

    for name in args.methods or cfg["methods"]:
        if lock and not lock.check(name, args.force):
            continue
        model = build_model(name, cfg, device)
        if lock:
            lock.mark(name, "started", device_name(device))
        seed = cfg["models"].get(name, {}).get("seed", 0)
        n = run_method(model, rows, jpeg_levels(cfg), paths,
                       out_path=lambda tag, content, i, name=name: paths.sr(name, tag, content, i),
                       seed_base=seed, runtime_log=paths.runtime_log(args.split, name), force=args.force)
        if lock:
            lock.mark(name, "finished", device_name(device))
        print(f"{name}: {n} images")
        del model


if __name__ == "__main__":
    main()
