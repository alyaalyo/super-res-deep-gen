"""Step 4: choose every free setting on the dev split, then freeze the config.

    python scripts/tune_dev.py

1. OCR configuration: grid over Tesseract page-segmentation mode, pre-OCR
   upscaling and the dictionary prior. Objective: mean CER over all dev text
   conditions (HR + every method x JPEG level, equally weighted), so the chosen
   reader does not favour any one method.
2. Face-match threshold: ArcFace impostor scores on dev (every HR/SR face of
   identity i vs the HR of identity j != i); threshold at the target
   false-match rate.
3. Diffusion option: ResShift fp16 autocast vs fp32 on a dev subset. fp16 is
   kept only if the fp16-vs-fp32 difference is small relative to the
   seed-to-seed variation (otherwise fp32 is used).

Writes configs/frozen.yaml and results/dev/*.csv.
"""

import argparse
import copy
import itertools
import time

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from srbench.config import DEFAULT_CONFIG, FROZEN_CONFIG, Paths, config_hash, jpeg_levels, jpeg_tag, load_yaml
from srbench.data.prepare import load_manifest
from srbench.evaluate import FaceCache
from srbench.imageio import imread
from srbench.metrics.face import threshold_at_fmr
from srbench.metrics.ocr import OCRConfig, cer, read_many
from srbench.runtime import pick_device

OCR_GRID = dict(psm=[4, 6], upscale=[1, 2, 3], dictionary=[True, False])


def dev_images(paths, cfg, rows, content):
    """(id, method, jpeg, path) for HR and every SR output of the dev split."""
    items = []
    for r in rows[rows.content == content].itertuples():
        items.append((r.id, "hr", "none", paths.hr(content, r.id)))
        for m in cfg["methods"]:
            for q in jpeg_levels(cfg):
                p = paths.sr(m, jpeg_tag(q), content, r.id)
                if not p.exists():
                    raise SystemExit(f"missing {p}; run scripts/run_sr.py --split dev first")
                items.append((r.id, m, jpeg_tag(q), p))
    return items


def tune_ocr(paths, cfg, rows, out_dir):
    import json

    gt = json.loads(paths.text_gt.read_text())
    items = dev_images(paths, cfg, rows, "text")
    cache = out_dir / "ocr_tuning_per_image.csv"
    grid = list(itertools.product(*OCR_GRID.values()))
    df = pd.read_csv(cache) if cache.exists() else None
    if df is not None and len(df) == len(grid) * len(items):
        print(f"  reusing {cache}")
    else:
        images = [imread(p) for *_, p in items]
        results = []
        for psm, up, dic in grid:
            oc = OCRConfig(psm=psm, upscale=up, dictionary=dic)
            t0 = time.time()
            texts = read_many(images, oc)
            for (image_id, method, tag, _), text in zip(items, texts):
                results.append(dict(psm=psm, upscale=up, dictionary=dic, id=image_id, method=method, jpeg=tag,
                                    cer=cer(text, gt[image_id])))
            print(f"  psm={psm} upscale={up} dictionary={dic}: {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(results)
        df.to_csv(cache, index=False)

    # mean per condition first, then over conditions: every (method, jpeg) weighs the same
    per_cond = df.groupby(["psm", "upscale", "dictionary", "method", "jpeg"]).cer.mean().reset_index()
    summary = per_cond.groupby(["psm", "upscale", "dictionary"]).cer.mean().rename("mean_cer").reset_index()
    hr = per_cond[per_cond.method == "hr"].set_index(["psm", "upscale", "dictionary"]).cer.rename("hr_cer")
    # ties (e.g. the dictionary switch, which Tesseract's LSTM engine ignores) go to Tesseract's defaults
    summary = (summary.join(hr, on=["psm", "upscale", "dictionary"]).round({"mean_cer": 6})
                      .sort_values(["mean_cer", "dictionary"], ascending=[True, False]))
    summary.to_csv(out_dir / "ocr_tuning.csv", index=False)
    print(summary.to_string(index=False))
    best = summary.iloc[0]
    return dict(psm=int(best.psm), upscale=int(best.upscale), dictionary=bool(best.dictionary)), summary


def tune_face_threshold(paths, cfg, rows, out_dir):
    faces = FaceCache(paths)
    items = dev_images(paths, cfg, rows, "faces")
    ids = sorted({i for i, *_ in items})
    hr_emb = {i: faces.hr_embedding(i) for i in ids}
    gallery = np.stack([hr_emb[i] for i in ids])
    genuine, impostor = [], []
    for image_id, method, tag, path in tqdm(items, desc="ArcFace dev"):
        e = faces.embed(imread(path), image_id)
        scores = gallery @ e
        for j, other in enumerate(ids):
            rec = dict(probe=image_id, gallery=other, method=method, jpeg=tag, score=float(scores[j]))
            (genuine if other == image_id else impostor).append(rec)
    gen = pd.DataFrame(genuine)
    imp = pd.DataFrame(impostor)
    imp = imp[(imp.method != "hr") | (imp.probe < imp.gallery)]   # count each HR-vs-HR pair once
    fmr = cfg["metrics"]["face"]["target_fmr"]
    thr = threshold_at_fmr(imp.score.values, fmr)
    pd.concat([gen.assign(pair="genuine"), imp.assign(pair="impostor")]).to_csv(out_dir / "face_scores.csv", index=False)
    print(f"  {len(imp)} impostor pairs, max {imp.score.max():.3f}; threshold at FMR {fmr}: {thr:.4f}")
    print(f"  dev genuine match rate at this threshold: {(gen[gen.method != 'hr'].score >= thr).mean():.3f}")
    return thr, len(imp)


def check_resshift_precision(cfg, paths, rows, device, n_per_content=4):
    """LPIPS(fp16 output, fp32 output) vs LPIPS(seed a, seed b) on dev images."""
    from srbench.metrics.perceptual import LPIPS
    from srbench.models import build_model, image_seed

    if device.type == "cpu":
        return True, {"note": "CPU: fp32 only"}
    lp = LPIPS(device)
    sub = rows.groupby("content").head(n_per_content)
    outs = {}
    for amp in (True, False):
        c = copy.deepcopy(cfg)
        c["models"]["resshift"]["amp"] = amp
        model = build_model("resshift", c, device)
        for r in tqdm(list(sub.itertuples()), desc=f"ResShift amp={amp}"):
            lr = imread(paths.lr("none", r.content, r.id))
            seeds = (image_seed(0, r.id), image_seed(1, r.id)) if amp else (image_seed(0, r.id),)
            for s in seeds:
                outs[(r.id, amp, s)] = model(lr, seed=s)
        del model
    prec, seed_var = [], []
    for r in sub.itertuples():
        s0, s1 = image_seed(0, r.id), image_seed(1, r.id)
        prec.append(lp(outs[(r.id, True, s0)], outs[(r.id, False, s0)]))
        seed_var.append(lp(outs[(r.id, True, s0)], outs[(r.id, True, s1)]))
    stats = dict(lpips_fp16_vs_fp32=float(np.mean(prec)), lpips_seed_vs_seed=float(np.mean(seed_var)))
    keep_amp = stats["lpips_fp16_vs_fp32"] < 0.25 * stats["lpips_seed_vs_seed"]
    print(f"  {stats} -> amp={keep_amp}")
    return keep_amp, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_yaml(DEFAULT_CONFIG)       # tuning always starts from the defaults
    paths = Paths(cfg)
    rows = load_manifest(paths, "dev")
    out_dir = paths.results / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    print("[1/3] OCR configuration")
    best_ocr, _ = tune_ocr(paths, cfg, rows, out_dir)
    print("[2/3] face-match threshold")
    thr, n_imp = tune_face_threshold(paths, cfg, rows, out_dir)
    print("[3/3] ResShift precision")
    amp, amp_stats = check_resshift_precision(cfg, paths, rows, device)

    frozen = copy.deepcopy(cfg)
    frozen["metrics"]["ocr"] = best_ocr
    frozen["metrics"]["face"]["threshold"] = round(thr, 4)
    frozen["models"]["resshift"]["amp"] = bool(amp)
    frozen["frozen"] = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": config_hash(frozen),
        "tuned_on": "dev split only",
        "ocr_grid": OCR_GRID,
        "face_impostor_pairs": n_imp,
        "resshift_precision_check": {k: round(v, 5) for k, v in amp_stats.items() if isinstance(v, float)},
    }
    if FROZEN_CONFIG.exists():
        old = load_yaml(FROZEN_CONFIG)
        if config_hash(old) != config_hash(frozen) and paths.test_lock.exists():
            raise SystemExit("frozen.yaml would change after test runs were recorded; refusing to overwrite.")
    with open(FROZEN_CONFIG, "w") as f:
        f.write("# Written by scripts/tune_dev.py; do not edit by hand. The test split runs with this file.\n")
        yaml.safe_dump(frozen, f, sort_keys=False)
    print(f"wrote {FROZEN_CONFIG} (hash {frozen['frozen']['config_hash']})")


if __name__ == "__main__":
    main()
