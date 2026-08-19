"""Fuel pump prices from Gasvaktin (open data, verified retail price changes).

Source: https://github.com/gasvaktin/gasvaktin — timestamped price-change
events per retailer since April 2016. We build a daily national price panel,
average it over Hagstofa's collection window (first half of the month), and
calibrate the resulting m/m change against the published fuel subindices.

Notes
- List prices (mean_bensin95 / mean_diesel), not discount prices: Hagstofa
  collects posted pump prices.
- The nowcast maps pump-price changes 1:1 into the subindex, so the Jan-2026
  excise overhaul does not break it (it broke Brent->pump equations, which we
  do not use here; see PLAN_1.md 1.4).
"""
from __future__ import annotations

import json

import pandas as pd
import requests

from .px_client import RAW_DIR

TRENDS_URL = "https://raw.githubusercontent.com/gasvaktin/gasvaktin/master/vaktin/trends.min.json"
COMPANIES = {
    "ao": "Atlantsolía", "co": "Costco", "dn": "Dælan", "n1": "N1", "ob": "ÓB",
    "ol": "Olís", "or": "Orkan", "ox": "Orkan X", "sk": "Skeljungur",
}
# Company weights: equal across the majors; Costco excluded (membership-only,
# not in Hagstofa's outlet set).
EXCLUDE = {"co"}


def fetch_trends(use_cache: bool = True) -> dict:
    cache = RAW_DIR / "gasvaktin_trends.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    r = requests.get(TRENDS_URL, timeout=120)
    r.raise_for_status()
    data = r.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def daily_panel(trends: dict, fuel: str = "bensin95") -> pd.DataFrame:
    """Daily price per company (forward-filled between change events)."""
    cols = {}
    for co, events in trends.items():
        if co in EXCLUDE:
            continue
        s = pd.Series(
            {pd.Timestamp(e["timestamp"]).normalize(): e[f"mean_{fuel}"] for e in events}
        ).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        cols[COMPANIES.get(co, co)] = s
    df = pd.DataFrame(cols)
    full_idx = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="D")
    return df.reindex(full_idx).ffill()


def collection_window_mean(daily: pd.Series, day_from: int = 1, day_to: int = 15) -> pd.Series:
    """Average price over Hagstofa's collection window (days 1-15) per month."""
    d = daily.dropna()
    in_window = (d.index.day >= day_from) & (d.index.day <= day_to)
    w = d[in_window]
    return w.groupby(w.index.to_period("M")).mean()


def national_price(fuel: str = "bensin95", use_cache: bool = True) -> pd.Series:
    """Equal-weighted national daily list price across retailers (ex Costco)."""
    panel = daily_panel(fetch_trends(use_cache=use_cache), fuel=fuel)
    return panel.mean(axis=1)


def scraped_mm(fuel: str = "bensin95", use_cache: bool = True) -> pd.Series:
    """m/m % change of the collection-window mean price — the nowcast input."""
    m = collection_window_mean(national_price(fuel, use_cache=use_cache))
    return (m.pct_change() * 100).rename(f"{fuel}_mm")
