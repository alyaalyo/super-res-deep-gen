"""Step 7: statistics, tables and figures from the test-split metrics.

    python scripts/analyze.py [--split test]

Writes results/tables/*.{csv,md}, results/figures/*.png and
results/summary.md (all numbers are mean ± half-width of the 95% bootstrap CI,
resampling images).
"""

import argparse
import json

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from srbench import figures
from srbench.config import Paths, load_config
from srbench.data.prepare import load_manifest
from srbench.evaluate import load_metrics
from srbench.figures import CONTENT_LABELS, JPEG_LABELS, JPEG_ORDER
from srbench.metrics.ocr import normalize
from srbench.models import LABELS
from srbench.stats import bootstrap_ci, fmt, paired_difference, summarize

METRICS = ["psnr", "ssim", "lpips", "musiq", "cer", "wer", "face_sim", "face_match"]
HEADERS = {"psnr": "PSNR ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓", "musiq": "MUSIQ ↑", "cer": "CER ↓",
           "wer": "WER ↓", "face_sim": "ArcFace sim ↑", "face_match": "match rate ↑"}
DIGITS = {"psnr": 2, "ssim": 3, "lpips": 3, "musiq": 1, "cer": 3, "wer": 3, "face_sim": 3, "face_match": 2}
CONTRASTS = [("realesrgan", "realesrnet"), ("resshift", "realesrnet"), ("realesrnet", "bicubic"),
             ("resshift", "realesrgan")]
TASK = {"objects": ["psnr", "ssim", "lpips", "musiq"],
        "faces": ["psnr", "ssim", "lpips", "musiq", "face_sim", "face_match"],
        "text": ["psnr", "ssim", "lpips", "musiq", "cer", "wer"]}


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def cell(summary, **kw):
    s = summary
    for k, v in kw.items():
        s = s[s[k] == v]
    return s


def main_tables(summary, methods, out_dir):
    parts = []
    for content, metrics in TASK.items():
        rows = []
        for tag in JPEG_ORDER:
            for m in (["hr"] if tag == "none" else []) + methods:
                r = {"JPEG": JPEG_LABELS[tag], "method": LABELS[m]}
                for met in metrics:
                    s = cell(summary, content=content, method=m, jpeg=tag, metric=met)
                    r[HEADERS[met]] = fmt(*s[["mean", "lo", "hi"]].iloc[0], DIGITS[met]) if len(s) else "–"
                rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(out_dir / f"main_{content}.csv", index=False)
        parts.append(f"### {CONTENT_LABELS[content]}\n\n{md_table(t)}\n")
    return "\n".join(parts)


def jpeg_sensitivity(df, methods, n_boot):
    """Is the drop from q30 -> q10 larger than from q50 -> q30? (equal 20-point steps)"""
    rows = []
    for content, metrics in TASK.items():
        for metric in [m for m in metrics if m in ("psnr", "lpips", "cer", "face_sim")]:
            for m in methods:
                w = df[(df.content == content) & (df.method == m)].pivot_table(index="id", columns="jpeg",
                                                                                values=metric)
                if not {"q50", "q30", "q10"} <= set(w.columns):
                    continue
                sign = 1 if metric in ("psnr", "face_sim") else -1       # positive = got worse
                d1 = sign * (w.q50 - w.q30)
                d2 = sign * (w.q30 - w.q10)
                d0 = sign * (w["none"] - w.q50)
                m1, m2 = bootstrap_ci(d1, n_boot), bootstrap_ci(d2, n_boot)
                diff = bootstrap_ci(d2 - d1, n_boot)
                rows.append(dict(content=content, metric=metric, method=m,
                                 loss_none_to_50=d0.mean(), loss_50_to_30=m1[0], loss_30_to_10=m2[0],
                                 ratio=m2[0] / m1[0] if m1[0] > 0 else np.nan,
                                 extra_loss_below_30=diff[0], lo=diff[1], hi=diff[2],
                                 sharper_below_30=bool(diff[1] > 0)))
    return pd.DataFrame(rows)


def ocr_edit_ops(paths, df_text, methods):
    """Split OCR errors into substitutions (a wrong but confident character),
    deletions (characters lost) and insertions, per reference character."""
    gt = json.loads(paths.text_gt.read_text())
    rows = []
    for m in ["hr"] + methods:
        f = paths.metrics("test", m).with_name(f"{m}_ocr.json")
        if not f.exists():
            continue
        for key, text in json.loads(f.read_text()).items():
            image_id, tag = key.split("/")
            ref, hyp = normalize(gt[image_id]), normalize(text)
            ops = Levenshtein.editops(hyp, ref)       # hyp -> ref: 'delete' = extra char in OCR output
            n = max(len(ref), 1)
            counts = {"replace": 0, "insert": 0, "delete": 0}
            for op in ops:
                counts[op.tag] += 1
            rows.append(dict(id=image_id, method=m, jpeg=tag, substitution=counts["replace"] / n,
                             missing=counts["insert"] / n, extra=counts["delete"] / n))
    ops = pd.DataFrame(rows).merge(df_text[["id", "kind", "font_size"]].drop_duplicates(), on="id")
    return ops


def pick_examples(df, methods):
    """Qualitative examples: a typical and a worst case for identity / text."""
    items = {}
    f = df[(df.content == "faces") & (df.jpeg == "none")].pivot_table(index="id", columns="method",
                                                                      values="face_sim")
    gen = [m for m in ("realesrgan", "resshift") if m in f]
    worst = (f["realesrnet"] - f[gen].min(axis=1)).idxmax() if gen and "realesrnet" in f else f.index[0]
    typical = (f["realesrnet"] - f[gen].min(axis=1)).sort_values().index[len(f) // 2] if gen else f.index[1]
    items["faces"] = [(typical, "faces", (176, 128, 256, 256)), (worst, "faces", (176, 128, 256, 256))]
    o = df[(df.content == "objects") & (df.jpeg == "none") & (df.method == "realesrgan")].sort_values("lpips")
    items["objects"] = [(o.id.iloc[len(o) // 2], "objects", (160, 160, 192, 192)),
                        (o.id.iloc[-1], "objects", (160, 160, 192, 192))]
    t = df[(df.content == "text") & (df.jpeg == "none")]
    items["text"] = [(sorted(t[t.id.str.contains("sans32_codes")].id)[0], "text", (20, 20, 160, 320)),
                     (sorted(t[t.id.str.contains("serif24_words")].id)[0], "text", (20, 20, 120, 320))]
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    args = ap.parse_args()

    cfg = load_config(args.split)
    paths = Paths(cfg)
    n_boot, ci = cfg["stats"]["n_boot"], cfg["stats"]["ci"]
    thr = cfg["metrics"]["face"]["threshold"]
    methods = [m for m in cfg["methods"]]
    tables, figs = paths.results / "tables", paths.results / "figures"
    if args.split == "dev":
        tables, figs = paths.results / "dev" / "tables", paths.results / "dev" / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    df = load_metrics(paths, args.split)
    manifest = load_manifest(paths, args.split)
    df = df.merge(manifest[["id", "kind", "font_size", "font"]], on="id", how="left")
    df["face_match"] = np.where(df.face_sim.notna(), (df.face_sim >= thr).astype(float), np.nan)
    methods = [m for m in methods if m in set(df.method)]

    summary = summarize(df, ["content", "method", "jpeg"], METRICS, n_boot, ci)
    summary.to_csv(tables / "summary_long.csv", index=False)

    text = df[df.content == "text"]
    font = summarize(text, ["kind", "font_size", "method", "jpeg"], ["cer"], n_boot, ci)
    font.to_csv(tables / "text_font_size.csv", index=False)

    diffs = pd.concat([paired_difference(df, a, b, ["content", "jpeg"], METRICS, n_boot, ci)
                       for a, b in CONTRASTS if a in methods and b in methods], ignore_index=True)
    diffs.to_csv(tables / "paired_differences.csv", index=False)

    sens = jpeg_sensitivity(df, methods, n_boot)
    sens.to_csv(tables / "jpeg_sensitivity.csv", index=False)

    ops = ocr_edit_ops(paths, text, methods) if args.split == "test" else pd.DataFrame()
    if len(ops):
        ops_sum = ops.groupby(["method", "jpeg", "kind"])[["substitution", "missing", "extra"]].mean().reset_index()
        ops_sum.to_csv(tables / "ocr_error_types.csv", index=False)

    runtime = []
    for f in sorted((paths.results / "runtime").glob("*.csv")):
        runtime.append(pd.read_csv(f).assign(source="benchmark"))
    for f in sorted((paths.outputs / "runtime").glob(f"{args.split}_*.csv")):
        runtime.append(pd.read_csv(f).assign(source=f"{args.split} run"))
    rt = pd.concat(runtime, ignore_index=True) if runtime else pd.DataFrame()
    if len(rt):
        rts = (rt.groupby(["source", "device", "method"])
                 .agg(n=("seconds", "size"), sec_mean=("seconds", "mean"), sec_median=("seconds", "median"),
                      peak_mb=("peak_mb", "max")).reset_index())
        rts.to_csv(tables / "runtime.csv", index=False)

    # figures
    figures.jpeg_sweep(summary, methods, figs / "jpeg_sweep.png")
    figures.perception_distortion(summary, methods, figs / "perception_distortion.png")
    figures.font_size_effect(font, methods, figs / "text_font_size.png")
    if thr is not None:
        figures.face_identity(df[df.content == "faces"], summary[summary.content == "faces"], methods, thr,
                              figs / "face_identity.png")
    figures.paired_contrasts(diffs, figs / "paired_differences.png")
    if len(rt):
        figures.runtime_bars(rt[rt.source == "benchmark"] if (rt.source == "benchmark").any() else rt, methods,
                             figs / "runtime.png")
    examples = pick_examples(df, methods)
    titles = {"objects": "Objects & scenes: median image (upper) and the one Real-ESRGAN renders worst by LPIPS (lower)",
              "faces": "Faces: median identity change (upper) and the largest one vs Real-ESRNet (lower)",
              "text": "Text: 8 px-high random codes (upper) and 6 px-high serif words (lower), in the LR input"}
    for content, items in examples.items():
        figures.qualitative(paths, items, methods, ["none", "q30"], figs / f"examples_{content}.png",
                            titles[content])

    # summary.md
    out = [f"# Results ({args.split} split)\n",
           f"Numbers are mean ± half-width of the {int(ci * 100)}% percentile-bootstrap CI over images "
           f"({n_boot} resamples). PSNR/SSIM on Y, {cfg['metrics']['crop_border']}-px border crop. "
           f"Face match threshold {thr} (dev, FMR {cfg['metrics']['face']['target_fmr']}). "
           f"OCR: Tesseract, {cfg['metrics']['ocr']}.\n",
           "## Main results\n", main_tables(summary, methods, tables)]

    out.append("\n## Paired differences (same images), no JPEG\n")
    d = diffs[(diffs.jpeg == "none")].copy()
    d["value"] = [fmt(r.mean, r.lo, r.hi, DIGITS[r.metric]) + (" *" if r.significant else "")
                  for r in d.itertuples()]
    for content, metrics in TASK.items():
        t = d[(d.content == content) & d.metric.isin(metrics)].pivot(index="contrast", columns="metric",
                                                                      values="value")
        t = t[[m for m in metrics if m in t]].rename(columns=HEADERS).reset_index()
        t["contrast"] = [" − ".join(LABELS[x] for x in c.split(" - ")) for c in t.contrast]
        out.append(f"### {CONTENT_LABELS[content]}\n\n{md_table(t)}\n")
    out.append("`*` = 95% CI excludes 0.\n")

    out.append("\n## JPEG sensitivity: is the loss from JPEG 30 → 10 larger than from 50 → 30?\n")
    out.append("Both steps are 20 quality points. `loss` is signed so that positive = worse "
               "(PSNR/ArcFace down, LPIPS/CER up). `extra` = mean per-image (loss 30→10 − loss 50→30) with 95% CI.\n")
    s = sens.copy()
    s["method"] = s.method.map(LABELS)
    s["extra (95% CI)"] = [f"{r.extra_loss_below_30:.3f} [{r.lo:.3f}, {r.hi:.3f}]" for r in s.itertuples()]
    out.append(md_table(s[["content", "metric", "method", "loss_none_to_50", "loss_50_to_30", "loss_30_to_10",
                           "ratio", "extra (95% CI)", "sharper_below_30"]].round(3)))
    if len(ops):
        out.append("\n\n## OCR error types (per reference character)\n")
        o = ops_sum[ops_sum.jpeg.isin(["none", "q30"])].copy()
        o["method"] = o.method.map(LABELS)
        out.append(md_table(o.round(3)))
    if len(rt):
        out.append("\n\n## Cost\n")
        out.append(md_table(rts.assign(method=rts.method.map(LABELS)).round(3)))
    (paths.results / ("summary.md" if args.split == "test" else "dev/summary.md")).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
