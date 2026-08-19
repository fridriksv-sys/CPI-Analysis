"""Imputed-rent (reiknuð húsaleiga, CP042) model — Phase 4.

CP042 is the largest CPI component (~21%) and the one with the shortest usable
sample: Hagstofa switched from user cost to rental equivalence (leiguígildi) in
June 2024, so pre-break history is a different data-generating process and is
excluded from estimation (PLAN_1.md §1.1, §4).

Model (chosen by OOS horse race, see scripts/explore_rent3.py):
  CP042 m/m[t] = 0.7 * EWMA_hl6(past CP042 m/m)          # stock persistence
               + 0.3 * (a + b * mean(HMS_rent m/m, lags 1..3))  # new-contract tilt

Rationale:
- The stock turns over slowly, so CP042 m/m is smooth and its own EWMA is a
  strong baseline (beats RW and AR(1) OOS on the post-break sample).
- HMS's leiguvísitala measures NEW-contract capital-area rents — a leading,
  noisier signal. It enters at lags 1..3 only, which are all published before a
  month-t forecast is made (HMS releases month t in t+1), so the model is
  vintage-clean: no look-ahead.
- The 0.3 tilt is deliberate shrinkage: with ~25 post-break months the HMS slope
  is weak (R^2~0.1), so it corrects the drift at turning points without letting
  a short, noisy regression dominate.

For h>=2 the HMS lags would themselves be forecasts, so the tilt is dropped and
the path carries the EWMA drift forward — the persistence the plan expects, and
the reason headline inflation is now less policy-rate sensitive than pre-2024
history implies (do not let a long-sample model re-impose the old rate beta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hms

BREAK = pd.Period("2024-07", "M")  # first m/m fully under rental equivalence
EWMA_HALFLIFE = 6
HMS_TILT = 0.30


def _ewma(vals: np.ndarray, halflife: float) -> float:
    a = 1 - 0.5 ** (1 / halflife)
    m = vals[0]
    for v in vals[1:]:
        m = a * v + (1 - a) * m
    return float(m)


def _hms_driver(hms_rent_mm: pd.Series) -> pd.Series:
    """Observable driver: mean of HMS rent m/m over lags 1..3."""
    return (hms_rent_mm.shift(1) + hms_rent_mm.shift(2) + hms_rent_mm.shift(3)) / 3


def fit_hms_tilt(cp042_mm: pd.Series, hms_rent_mm: pd.Series,
                 train_end: pd.Period | None = None) -> tuple[float, float] | None:
    """OLS of post-break CP042 m/m on the lag-1..3 HMS driver. None if too short."""
    drv = _hms_driver(hms_rent_mm)
    df = pd.concat([cp042_mm.rename("y"), drv.rename("x")], axis=1)
    df = df[df.index >= BREAK].dropna()
    if train_end is not None:
        df = df[df.index <= train_end]
    if len(df) < 8 or df.x.var() == 0:
        return None
    b, a = np.polyfit(df.x, df.y, 1)
    return float(a), float(b)


def forecast_cp042(cp042_mm: pd.Series, hms_rent_mm: pd.Series | None,
                   jump_off: pd.Period, horizons: int = 12) -> pd.Series:
    """Forecast CP042 m/m for jump_off+1 .. jump_off+horizons.

    cp042_mm: history of CP042 m/m (%) through jump_off.
    hms_rent_mm: history of HMS rent m/m (%) through >= jump_off (published with a
                 lag, but lags 1..3 relative to the h=1 target are available).
    """
    hist = cp042_mm[cp042_mm.index >= BREAK].dropna()
    if len(hist) < 3:
        hist = cp042_mm.dropna().tail(12)  # fallback before enough post-break data
    ewma_drift = _ewma(hist.values, EWMA_HALFLIFE)

    tilt = None
    if hms_rent_mm is not None:
        coef = fit_hms_tilt(cp042_mm, hms_rent_mm, train_end=jump_off)
        if coef is not None:
            drv = _hms_driver(hms_rent_mm)
            # driver value for the h=1 target month uses HMS at lags 1..3 = months
            # jump_off, jump_off-1, jump_off-2, all observed.
            x1 = (hms_rent_mm.get(jump_off, np.nan)
                  + hms_rent_mm.get(jump_off - 1, np.nan)
                  + hms_rent_mm.get(jump_off - 2, np.nan)) / 3
            if not np.isnan(x1):
                a, b = coef
                tilt = a + b * x1

    months = pd.period_range(jump_off + 1, periods=horizons, freq="M")
    out = []
    for h, m in enumerate(months, start=1):
        if h == 1 and tilt is not None:
            out.append((1 - HMS_TILT) * ewma_drift + HMS_TILT * tilt)
        else:
            out.append(ewma_drift)  # h>=2: HMS lags unknown -> pure persistence
    return pd.Series(out, index=months, name="CP042")


def load_cp042_history(sub_spliced: pd.DataFrame) -> pd.Series:
    """CP042 m/m (%) history from the spliced subindex levels."""
    lvl = sub_spliced[sub_spliced.code == "CP042"].set_index("manudur").visitala
    return (lvl.pct_change() * 100).rename("CP042_mm")


def hms_rent_mm(use_cache: bool = True) -> pd.Series:
    """HMS combined rental index m/m (%)."""
    return (hms.load_rent(use_cache=use_cache).leiguvisitala.pct_change() * 100).rename("hms_rent_mm")
