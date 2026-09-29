"""Step 6: how much do ResShift outputs depend on the random seed?

    python scripts/seed_experiment.py

Runs ResShift with several seeds on a fixed test subset (seed 0 = the main
test run, reused) and measures (a) pixel-level variation between seeds,
(b) the spread of every metric across seeds, (c) how often a task decision
flips: face match / no match, and the OCR transcription.

Writes results/seed_experiment/{per_seed.csv, summary.csv, summary.md} and
results/figures/seed_variation.png.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from srbench.config import Paths, jpeg_tag, load_config
from srbench.data.prepare import load_manifest
from srbench.evaluate import Evaluator
from srbench.imageio import imread
from srbench.metrics.fidelity import rgb2y
from srbench.models import build_model
from srbench.pipeline import TestLock, run_method
from srbench.runtime import pick_device


def subset(rows: pd.DataFrame, per_content: int, seed: int) -> pd.DataFrame:
    return (rows.groupby("content").sample(n=per_content, random_state=seed)
                .sort_values(["content", "id"]).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config("test")
    paths = Paths(cfg)
    lock = TestLock(paths, cfg)
    if not lock.state.get("resshift", {}).get("finished"):
        raise SystemExit("run the main test split first: scripts/run_sr.py --split test")
    se = cfg["seed_experiment"]
    main_seed = cfg["models"]["resshift"]["seed"]
    jpegs = [None if str(q) == "none" else int(q) for q in se["jpeg_qualities"]]
    rows = subset(load_manifest(paths, "test"), se["per_content"], cfg["seed"])
    out_root = paths.outputs / "seeds" / "resshift"

    def seed_path(s):
        if s == main_seed:
            return lambda tag, content, i: paths.sr("resshift", tag, content, i)
        return lambda tag, content, i: out_root / f"s{s}" / tag / content / f"{i}.png"

    device = pick_device(args.device)
    model = build_model("resshift", cfg, device)
    for s in se["seeds"]:
        if s != main_seed:
            run_method(model, rows, jpegs, paths, seed_path(s), seed_base=s)
    del model

    # metrics for every seed
    evaluator = Evaluator(cfg, paths, device)
    frames = []
    for s in se["seeds"]:
        fn = seed_path(s)
        df, _ = evaluator.evaluate("resshift", rows, jpegs, image_path=lambda m, t, c, i, fn=fn: fn(t, c, i))
        frames.append(df.assign(seed=s))
    per_seed = pd.concat(frames, ignore_index=True)
    thr = cfg["metrics"]["face"]["threshold"]
    per_seed["face_match"] = np.where(per_seed.face_sim.notna(), per_seed.face_sim >= thr, np.nan)

    # pixel variation and output diversity between seeds
    lp = evaluator.lpips
    pix = []
    for r in rows.itertuples():
        for q in jpegs:
            tag = jpeg_tag(q)
            outs = [imread(seed_path(s)(tag, r.content, r.id)) for s in se["seeds"]]
            ys = np.stack([rgb2y(o) for o in outs])
            pairs = list(itertools.combinations(range(len(outs)), 2))
            pix.append(dict(id=r.id, content=r.content, jpeg=tag,
                            pixel_std=float(ys.std(axis=0, ddof=1).mean()),
                            lpips_between_seeds=float(np.mean([lp(outs[i], outs[j]) for i, j in pairs]))))
    pix = pd.DataFrame(pix)

    metrics = [m for m in ["psnr", "ssim", "lpips", "musiq", "face_sim", "cer"] if m in per_seed]
    spread = per_seed.groupby(["id", "content", "jpeg"])[metrics].agg(lambda v: v.std(ddof=1)).reset_index()
    flips = per_seed.groupby(["id", "content", "jpeg"]).agg(
        match_flip=("face_match", lambda v: float(v.nunique() > 1) if v.notna().any() else np.nan),
        cer_range=("cer", lambda v: v.max() - v.min() if v.notna().any() else np.nan),
    ).reset_index()

    out_dir = paths.results / "seed_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "per_seed.csv", index=False)
    per_image = pix.merge(spread, on=["id", "content", "jpeg"]).merge(flips, on=["id", "content", "jpeg"])
    per_image.to_csv(out_dir / "per_image.csv", index=False)
    cols = ["pixel_std", "lpips_between_seeds"] + [m for m in metrics] + ["match_flip", "cer_range"]
    summary = per_image.groupby(["content", "jpeg"])[cols].mean().reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.md", "w") as f:
        f.write(f"# ResShift seed experiment\n\n{len(rows)} test images ({se['per_content']} per content) x "
                f"{len(jpegs)} JPEG levels x {len(se['seeds'])} seeds. Metric columns are the across-seed "
                "standard deviation, averaged over images. `match_flip` = fraction of faces whose "
                "match/no-match decision differs between seeds; `cer_range` = max - min CER across seeds.\n\n")
        f.write(summary.round(4).to_markdown(index=False))
        f.write("\n")
    json.dump({"rows": rows.id.tolist()}, open(out_dir / "subset.json", "w"))
    print(summary.round(4).to_string(index=False))

    from srbench.figures import seed_variation_figure
    seed_variation_figure(paths, rows, se["seeds"], seed_path)


if __name__ == "__main__":
    main()
