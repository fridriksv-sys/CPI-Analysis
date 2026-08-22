"""Top-down headline model (Phase 6).

A small, inspectable unobserved-components-style model on headline VNV m/m:

    mm_t = s[month] + mu_t + e_t,   e_t = phi * e_{t-1} + eps

- s[month]: fixed seasonal factors (deviation of each calendar month's mean
  m/m from the overall mean), estimated 2015-.
- mu_t: a slow-moving trend. Its forecast GLIDES from the recent average toward
  an anchor at the long end. Per PLAN_1.md §6 the anchor sits between the 2.5%
  Seðlabanki target and breakeven-implied inflation; breakevens need the RIKS/RIKB
  market feed (Phase 7), so the default anchor is the target and `anchor_yoy` is
  exposed for when the breakeven number is available.
- e_t: AR(1) cyclical deviation, mean-reverting to 0.

This is the aggregate forecast that MinT reconciles the bottom-up path against;
at long horizons it dominates (component errors compound), which is the point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SAMPLE_START = pd.Period("2015-01", "M")
TREND_WINDOW = 36        # months averaged for the recent trend
TREND_HALFLIFE = 18      # months for the glide toward the anchor
TARGET_YOY = 2.5         # Seðlabanki inflation target (%)


def _target_monthly(anchor_yoy: float) -> float:
    return (1 + anchor_yoy / 100) ** (1 / 12) * 100 - 100


@dataclass
class TopDownFit:
    seasonal: pd.Series      # month -> deviation factor (pp)
    mu_recent: float         # recent trend (monthly, pp)
    phi: float               # AR(1) on the cyclical deviation
    e_last: float            # last cyclical deviation
    resid_sd: float
    last_month: pd.Period


def fit(mm: pd.Series) -> TopDownFit:
    m = mm.dropna()
    m = m[m.index >= SAMPLE_START]
    overall = m.mean()
    seasonal = m.groupby(m.index.month).mean() - overall
    de = m - seasonal.reindex(m.index.month).to_numpy() - overall
    x, y = de.shift(1).dropna(), de.iloc[1:]
    x, y = x.align(y, join="inner")
    phi = float(np.clip((x * y).sum() / (x * x).sum(), 0.0, 0.9)) if x.var() > 0 else 0.0
    resid = y - phi * x
    mu_recent = m.tail(TREND_WINDOW).mean()
    e_last = float(de.iloc[-1])
    return TopDownFit(seasonal, float(mu_recent), phi, e_last, float(resid.std()), m.index[-1])


def forecast(f: TopDownFit, horizons: int = 12, anchor_yoy: float = TARGET_YOY) -> pd.Series:
    """Top-down headline m/m path for last_month+1 .. +horizons."""
    mu_anchor = _target_monthly(anchor_yoy)
    rho = 0.5 ** (1 / TREND_HALFLIFE)
    months = pd.period_range(f.last_month + 1, periods=horizons, freq="M")
    out, e = [], f.e_last
    for h, mth in enumerate(months, start=1):
        e = f.phi * e
        mu_h = mu_anchor + (f.mu_recent - mu_anchor) * rho ** h
        out.append(f.seasonal.get(mth.month, 0.0) + mu_h + e)
    return pd.Series(out, index=months, name="topdown_mm")


def error_sd_by_horizon(f: TopDownFit, horizons: int = 12) -> pd.Series:
    """Approx cumulative-innovation SD of the m/m forecast at each horizon.

    AR(1) h-step variance = sigma^2 * (1 - phi^{2h}) / (1 - phi^2).
    """
    var1 = f.resid_sd ** 2
    hs = np.arange(1, horizons + 1)
    if f.phi >= 1:
        v = var1 * hs
    else:
        v = var1 * (1 - f.phi ** (2 * hs)) / (1 - f.phi ** 2)
    months = pd.period_range(f.last_month + 1, periods=horizons, freq="M")
    return pd.Series(np.sqrt(v), index=months, name="topdown_sd")
