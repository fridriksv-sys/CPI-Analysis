"""h=1 nowcast layer: replace model forecasts with OBSERVED inputs where we
have them (PLAN_1.md Phase 3).

Currently wired observables:
- CP0722 eldsneyti: Gasvaktin pump prices averaged over the collection window,
  passed through a calibration regression (published subindex on scraped m/m)
  estimated on 2016- history of the old IS0722 subindex.

Collecting, calibration pending (need >=2 collection windows of scraped history
before the observable can override the model forecast):
- CP0733 airfares: Icelandair KEF-origin lowest fares (see airfares.py)
- CP011x groceries: Kronan price snapshots (see groceries.py)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import airfares, fuel, ingest


@dataclass
class FuelCalibration:
    alpha: float
    beta: float
    resid_sd: float
    n_obs: int


def calibrate_fuel(train_end: pd.Period | None = None, use_cache: bool = True) -> FuelCalibration:
    """OLS of published IS0722 (eldsneyti) m/m on scraped collection-window m/m.

    Long-sample (2016-) calibration at the combined-fuel level: the bensin/diesel
    split only exists in the published data from 2025, too short to calibrate on.
    Excludes 2026-01/02 (excise-overhaul + the documented wrong-price incident).
    """
    bensin = fuel.scraped_mm("bensin95", use_cache=use_cache)
    diesel = fuel.scraped_mm("diesel", use_cache=use_cache)
    mix = (2 / 3) * bensin + (1 / 3) * diesel

    old = ingest.load_panel_old()
    pub = old[old.code == "IS0722"].set_index("manudur").manadarbreyting
    df = pd.concat([mix.rename("scraped"), pub.rename("pub")], axis=1).dropna()
    df = df[df.index >= "2016-06"]
    df = df.drop([pd.Period("2026-01", "M"), pd.Period("2026-02", "M")], errors="ignore")
    if train_end is not None:
        df = df[df.index <= train_end]

    x, y = df.scraped, df.pub
    beta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    alpha = y.mean() - beta * x.mean()
    resid = y - (alpha + beta * x)
    return FuelCalibration(float(alpha), float(beta), float(resid.std()), len(df))


def fuel_nowcast(month: pd.Period, cal: FuelCalibration | None = None,
                 use_cache: bool = True) -> dict[str, float]:
    """Calibrated m/m nowcast for CP0722 in `month` from pump prices."""
    if cal is None:
        cal = calibrate_fuel(use_cache=use_cache)
    bensin = fuel.scraped_mm("bensin95", use_cache=use_cache)
    diesel = fuel.scraped_mm("diesel", use_cache=use_cache)
    mix = (2 / 3) * bensin + (1 / 3) * diesel
    if month not in mix.index:
        raise ValueError(f"no scraped fuel data for {month}")
    return {"CP0722": cal.alpha + cal.beta * float(mix[month])}


@dataclass
class AirfareCalibration:
    alpha: float
    beta: float
    resid_sd: float
    n_obs: int


def calibrate_airfares(train_end: pd.Period | None = None) -> AirfareCalibration | None:
    """OLS of published CP073 (passenger transport) m/m on scraped airfare m/m.

    CP073 is the forecast component; international air (CP07332) is its dominant
    mover, so regressing the component itself on the scraped air index lets the
    slope absorb the sub-group share. Returns None until the scraped history
    spans enough overlapping months (>= 6) - no synthetic priors, the override
    stays inactive until real calibration data exists.
    """
    scraped = airfares.airfare_index_mm()
    if scraped.empty:
        return None
    sub = ingest.load_sub_new()
    pub = sub[sub.code == "CP073"].set_index("manudur").manadarbreyting
    df = pd.concat([scraped.rename("scraped"), pub.rename("pub")], axis=1).dropna()
    if train_end is not None:
        df = df[df.index <= train_end]
    if len(df) < 6:
        return None
    x, y = df.scraped, df.pub
    beta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    alpha = y.mean() - beta * x.mean()
    resid = y - (alpha + beta * x)
    return AirfareCalibration(float(alpha), float(beta), float(resid.std()), len(df))


def airfare_nowcast(month: pd.Period, cal: AirfareCalibration | None = None) -> dict[str, float]:
    """Calibrated m/m nowcast for CP073 in `month`. Empty dict if not yet calibrated."""
    if cal is None:
        cal = calibrate_airfares()
    if cal is None:
        return {}
    scraped = airfares.airfare_index_mm()
    if month not in scraped.index:
        return {}
    return {"CP073": cal.alpha + cal.beta * float(scraped[month])}


def apply_observables(fcst_mm: pd.DataFrame, observed: dict[str, float]) -> pd.DataFrame:
    """Override the FIRST forecast month's components with observed values."""
    out = fcst_mm.copy()
    first = out.index[0]
    for code, val in observed.items():
        if code in out.columns:
            out.loc[first, code] = val
    return out
