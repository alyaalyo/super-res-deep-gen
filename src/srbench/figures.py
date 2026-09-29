"""Figures for the report (static PNG, matplotlib).

Colour is assigned by method and never changes between figures: Bicubic is a
neutral dark grey (the baseline), the three learned models take the first
three categorical hues. Each method also has its own marker, so identity never
depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Paths  # noqa: E402
from .imageio import imread  # noqa: E402
from .models import LABELS  # noqa: E402

COLORS = {"bicubic": "#52514e", "realesrnet": "#2a78d6", "realesrgan": "#eb6834", "resshift": "#1baf7a",
          "sd_upscaler": "#4a3aa7"}
MARKERS = {"bicubic": "s", "realesrnet": "o", "realesrgan": "D", "resshift": "^", "sd_upscaler": "v"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
JPEG_ORDER = ["none", "q50", "q30", "q10"]
JPEG_LABELS = {"none": "none", "q50": "50", "q30": "30", "q10": "10"}
CONTENT_LABELS = {"objects": "Objects & scenes", "faces": "Faces", "text": "Text"}
METRIC_LABELS = {
    "psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓", "musiq": "MUSIQ ↑",
    "cer": "OCR character error rate ↓", "face_sim": "ArcFace similarity to HR ↑",
    "face_match": "Identity match rate ↑",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 9, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "axes.axisbelow": True,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _legend(fig, methods, extra=None, ncol=None):
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], markersize=7, markeredgecolor=SURFACE,
                          markeredgewidth=1.5, label=LABELS[m]) for m in methods]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), bbox_to_anchor=(0.5, 1.02))


def _cell(summary: pd.DataFrame, **kw) -> pd.DataFrame:
    s = summary
    for k, v in kw.items():
        s = s[s[k] == v]
    return s


def jpeg_sweep(summary: pd.DataFrame, methods: list[str], out: Path):
    """Rows: content; columns: metrics; x: JPEG quality; one line per method with 95% CI."""
    panels = {
        "objects": ["psnr", "lpips", "musiq"],
        "faces": ["psnr", "lpips", "face_sim"],
        "text": ["psnr", "lpips", "cer"],
    }
    style()
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
    x = np.arange(len(JPEG_ORDER))
    for i, (content, metrics) in enumerate(panels.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for m in methods:
                s = _cell(summary, content=content, method=m, metric=metric).set_index("jpeg").reindex(JPEG_ORDER)
                ax.fill_between(x, s.lo, s.hi, color=COLORS[m], alpha=0.12, linewidth=0)
                ax.plot(x, s["mean"], color=COLORS[m], marker=MARKERS[m], markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            hr = _cell(summary, content=content, method="hr", metric=metric)
            if metric in ("musiq", "cer") and len(hr):
                ax.axhline(hr["mean"].iloc[0], color=INK2, linewidth=1)
                ax.annotate("HR", (x[0], hr["mean"].iloc[0]), xytext=(0, 3), textcoords="offset points",
                            va="bottom", ha="left", fontsize=8, color=INK2)
            ax.set_xticks(x, [JPEG_LABELS[t] for t in JPEG_ORDER])
            ax.set_title(f"{CONTENT_LABELS[content]}: {METRIC_LABELS[metric]}", loc="left", fontsize=10)
            if i == 2:
                ax.set_xlabel("JPEG quality of the LR input")
    _legend(fig, methods)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out)


def perception_distortion(summary: pd.DataFrame, methods: list[str], out: Path):
    """PSNR (distortion) against LPIPS (perceptual distance); arrows follow JPEG none -> 10."""
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, content in zip(axes, ["objects", "faces", "text"]):
        for m in methods:
            p = _cell(summary, content=content, method=m, metric="psnr").set_index("jpeg").reindex(JPEG_ORDER)
            l = _cell(summary, content=content, method=m, metric="lpips").set_index("jpeg").reindex(JPEG_ORDER)
            ax.plot(p["mean"], l["mean"], color=COLORS[m], linewidth=1.2, alpha=0.6)
            ax.scatter(p["mean"].iloc[1:], l["mean"].iloc[1:], color=COLORS[m], marker=MARKERS[m], s=30,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, alpha=0.55)
            ax.scatter(p["mean"].iloc[:1], l["mean"].iloc[:1], color=COLORS[m], marker=MARKERS[m], s=70,
                       edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.invert_yaxis()
        ax.set_title(CONTENT_LABELS[content], loc="left")
        ax.set_xlabel("PSNR (dB) →  more faithful")
        ax.set_ylabel("LPIPS (inverted) ↑ perceptually closer")
    extra = [plt.Line2D([], [], color=INK2, marker="o", markersize=8, linestyle="", label="no JPEG (large marker)"),
             plt.Line2D([], [], color=INK2, marker="o", markersize=5, alpha=0.55, linestyle="",
                        label="JPEG 50 → 30 → 10")]
    _legend(fig, methods, extra, ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out)


def font_size_effect(summary: pd.DataFrame, methods: list[str], out: Path, jpegs=("none", "q30")):
    """OCR CER against font size, words vs random codes, for two JPEG levels."""
    style()
    kinds = ["words", "codes"]
    fig, axes = plt.subplots(len(jpegs), 2, figsize=(10, 3.6 * len(jpegs)), sharey=True, squeeze=False)
    for i, tag in enumerate(jpegs):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            for m in ["hr"] + methods:
                s = _cell(summary, kind=kind, method=m, jpeg="none" if m == "hr" else tag).sort_values("font_size")
                if s.empty:
                    continue
                if m == "hr":
                    ax.plot(s.font_size, s["mean"], color=INK2, linewidth=1)
                    ax.annotate("HR", (s.font_size.iloc[-1], s["mean"].iloc[-1]), xytext=(4, 0),
                                textcoords="offset points", fontsize=8, color=INK2, va="center")
                    continue
                ax.fill_between(s.font_size, s.lo, s.hi, color=COLORS[m], alpha=0.12, linewidth=0)
                ax.plot(s.font_size, s["mean"], color=COLORS[m], marker=MARKERS[m], markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            sizes = sorted(summary.font_size.dropna().unique())
            ax.set_xticks(sizes, [f"{int(s)}\n({int(s) // 4} px LR)" for s in sizes])
            ax.set_title(f"{kind.capitalize()}, JPEG {JPEG_LABELS[tag]}", loc="left")
            if j == 0:
                ax.set_ylabel("OCR character error rate ↓")
            if i == len(jpegs) - 1:
                ax.set_xlabel("font size in HR pixels")
    _legend(fig, methods)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out)


def face_identity(df: pd.DataFrame, summary: pd.DataFrame, methods: list[str], threshold: float, out: Path):
    """Per-image ArcFace similarity (strip + mean) and, unless it is saturated,
    the match rate with its 95% CI."""
    style()
    rates = summary[(summary.metric == "face_match") & summary.method.isin(methods)]
    saturated = len(rates) and rates["mean"].min() >= 0.999
    if saturated:
        fig, a1 = plt.subplots(figsize=(9, 4.4))
        a2 = None
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    width = 0.8 / len(methods)
    rng = np.random.default_rng(0)
    for k, m in enumerate(methods):
        for i, tag in enumerate(JPEG_ORDER):
            v = df[(df.method == m) & (df.jpeg == tag)].face_sim.dropna().values
            xpos = i + (k - (len(methods) - 1) / 2) * width
            a1.scatter(xpos + rng.uniform(-width / 4, width / 4, len(v)), v, s=5, color=COLORS[m], alpha=0.35,
                       linewidth=0)
            a1.plot([xpos - width / 3, xpos + width / 3], [v.mean()] * 2, color=COLORS[m], linewidth=2.5)
            s = _cell(summary, method=m, jpeg=tag, metric="face_match")
            if a2 is not None and len(s):
                a2.bar(xpos, s["mean"].iloc[0], width=width - 0.03, color=COLORS[m])
                a2.errorbar(xpos, s["mean"].iloc[0], yerr=[[s["mean"].iloc[0] - s.lo.iloc[0]],
                                                            [s.hi.iloc[0] - s["mean"].iloc[0]]],
                            color=INK2, linewidth=1, capsize=0)
    a1.axhline(threshold, color=INK2, linewidth=1)
    a1.annotate(f"match threshold {threshold:.3f} (dev, FMR 0.1%)", (len(JPEG_ORDER) - 0.5, threshold),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    for ax in (a1, a2):
        if ax is None:
            continue
        ax.set_xticks(range(len(JPEG_ORDER)), [JPEG_LABELS[t] for t in JPEG_ORDER])
        ax.set_xlabel("JPEG quality of the LR input")
    a1.set_ylabel("ArcFace cosine similarity to HR")
    a1.set_ylim(min(threshold, df.face_sim.min()) - 0.05, 1.01)
    title = "Identity similarity per image (bar = mean)"
    if saturated:
        title += "\nEvery face stays above the match threshold: match rate = 100% for all methods and JPEG levels"
    a1.set_title(title, loc="left")
    if a2 is not None:
        a2.set_ylabel("share of faces matched to their HR")
        a2.set_ylim(0, 1.05)
        a2.set_title("Identity match rate (95% CI)", loc="left")
    _legend(fig, methods)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out)


def paired_contrasts(diffs: pd.DataFrame, out: Path, metrics=("psnr", "lpips", "musiq", "cer", "face_sim")):
    """Forest plot of paired differences (A - B) per content at JPEG none."""
    style()
    d = diffs[(diffs.jpeg == "none") & diffs.metric.isin(metrics)]
    contrasts = list(dict.fromkeys(d.contrast))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 0.45 * len(contrasts) * 3 + 1.2),
                             sharey=True)
    ylabels = []
    for content in ["objects", "faces", "text"]:
        for c in contrasts:
            ylabels.append((content, c))
    for ax, metric in zip(axes, metrics):
        for y, (content, c) in enumerate(ylabels):
            s = d[(d.content == content) & (d.contrast == c) & (d.metric == metric)]
            if s.empty:
                continue
            a = c.split(" - ")[0]
            ax.plot([s.lo.iloc[0], s.hi.iloc[0]], [y, y], color=COLORS.get(a, INK2), linewidth=2)
            ax.scatter(s["mean"], [y], color=COLORS.get(a, INK2), marker=MARKERS.get(a, "o"), s=36, zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(METRIC_LABELS[metric].replace("OCR character error rate", "CER"), loc="left", fontsize=9)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(ylabels)),
                       [f"{CONTENT_LABELS[ct]}: {LABELS[c.split(' - ')[0]]} − {LABELS[c.split(' - ')[1]]}"
                        for ct, c in ylabels])
    fig.suptitle("Paired differences on the same images, no JPEG (mean, 95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out)


def runtime_bars(rt: pd.DataFrame, methods: list[str], out: Path):
    style()
    devices = list(dict.fromkeys(rt.device))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.6 * len(devices) * len(methods) / 2 + 2.2))
    for ax, (col, label) in zip(axes, [("seconds", "seconds per image (128 → 512)"),
                                       ("peak_mb", "peak accelerator memory (MB)")]):
        ylabels, y = [], 0
        for dev in devices:
            for m in methods:
                s = rt[(rt.device == dev) & (rt.method == m)]
                if s.empty or s[col].isna().all():
                    continue
                v = s[col].mean() if col == "seconds" else s[col].max()
                ax.barh(y, v, height=0.6, color=COLORS[m])
                ax.annotate(f"{v:.2f}" if col == "seconds" else f"{v:.0f}", (v, y), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK)
                ylabels.append(f"{LABELS[m]} ({dev})")
                y += 1
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        if col == "seconds":
            ax.set_xscale("log")
    fig.tight_layout()
    _save(fig, out)


def qualitative(paths: Paths, items: list[tuple[str, str, tuple[int, int, int, int]]], methods: list[str],
                jpegs: list[str], out: Path, title: str):
    """Rows = (image, JPEG level); columns = LR (nearest x4), HR, methods. Crops given as (y, x, h, w)."""
    style()
    rows = [(cid, content, box, tag) for cid, content, box in items for tag in jpegs]
    cols = ["lr", "hr"] + methods
    cw = 1.9
    ratios = [box[2] / box[3] for _, _, box, _ in rows]
    fig = plt.figure(figsize=(cw * len(cols), cw * sum(ratios) + 0.9))
    gs = fig.add_gridspec(len(rows), len(cols), height_ratios=ratios, hspace=0.06, wspace=0.03,
                          left=0.03, right=0.995, bottom=0.01, top=1 - 0.75 / (cw * sum(ratios) + 0.9))
    for i, (cid, content, (y, x, h, w), tag) in enumerate(rows):
        for j, col in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            if col == "hr":
                img = imread(paths.hr(content, cid))
            elif col == "lr":
                img = np.kron(imread(paths.lr(tag, content, cid)), np.ones((4, 4, 1), dtype=np.uint8))
            else:
                img = imread(paths.sr(col, tag, content, cid))
            ax.imshow(img[y:y + h, x:x + w], interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title({"lr": "LR input (x4 nearest)", "hr": "HR"}.get(col, LABELS.get(col, col)), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"JPEG {JPEG_LABELS[tag]}", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    _save(fig, out)


def seed_variation_figure(paths: Paths, rows: pd.DataFrame, seeds: list[int], seed_path, tag: str = "none"):
    """One image per content: HR crop, two seeds, per-pixel std (Y) across all seeds."""
    from .metrics.fidelity import rgb2y

    style()
    picks = rows.groupby("content").head(1)
    fig = plt.figure(figsize=(9.6, 2.4 * len(picks) + 0.5))
    gs = fig.add_gridspec(len(picks), 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.05, hspace=0.08)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(len(picks))])
    for i, r in enumerate(picks.itertuples()):
        outs = [imread(seed_path(s)(tag, r.content, r.id)) for s in seeds]
        std = np.stack([rgb2y(o) for o in outs]).std(axis=0, ddof=1)
        panels = [(imread(paths.hr(r.content, r.id)), "HR"), (outs[0], f"seed {seeds[0]}"),
                  (outs[1], f"seed {seeds[1]}"), (std, f"pixel std over {len(seeds)} seeds")]
        for j, (img, t) in enumerate(panels):
            ax = axes[i, j]
            c = slice(128, 384)
            if img.ndim == 2:
                im = ax.imshow(img[c, c], cmap="Blues", vmin=0, vmax=20)
            else:
                ax.imshow(img[c, c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(CONTENT_LABELS[r.content], fontsize=9)
    fig.colorbar(im, cax=fig.add_subplot(gs[:, 4]), label="std of luma across seeds (0-255)")
    fig.suptitle(f"ResShift: variation between random seeds (JPEG {JPEG_LABELS[tag]})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    _save(fig, paths.results / "figures" / "seed_variation.png")
