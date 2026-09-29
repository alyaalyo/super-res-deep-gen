"""Means with percentile-bootstrap 95% confidence intervals.

The resampling unit is the image. Paired differences (method A - method B on
the same images) are bootstrapped over images too, which is much tighter than
comparing two independent CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(x, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str], n_boot: int = 10_000,
              ci: float = 0.95) -> pd.DataFrame:
    """Long table: by..., metric, n, mean, lo, hi."""
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            if m not in g or g[m].notna().sum() == 0:
                continue
            vals = g[m].values
            mean, lo, hi = bootstrap_ci(vals, n_boot, ci)
            rows.append(dict(zip(by, key), metric=m, n=int(np.isfinite(vals.astype(float)).sum()),
                             mean=mean, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def paired_difference(df: pd.DataFrame, a: str, b: str, by: list[str], metrics: list[str],
                      n_boot: int = 10_000, ci: float = 0.95) -> pd.DataFrame:
    """Mean of (method a - method b) per image, with a bootstrap CI over images."""
    wide = df[df.method.isin([a, b])].pivot_table(index=["id", *by], columns="method", values=metrics)
    rows = []
    for key, g in wide.groupby(level=list(range(1, 1 + len(by))), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            if (m, a) not in g or (m, b) not in g:
                continue
            d = (g[(m, a)] - g[(m, b)]).values
            mean, lo, hi = bootstrap_ci(d, n_boot, ci)
            rows.append(dict(zip(by, key), contrast=f"{a} - {b}", metric=m, n=int(np.isfinite(d).sum()),
                             mean=mean, lo=lo, hi=hi, significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def fmt(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return "–"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {(hi - lo) / 2:.{digits}f}"
