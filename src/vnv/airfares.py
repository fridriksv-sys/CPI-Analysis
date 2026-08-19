"""Airfares (CP0733 / CP07332): the highest-leverage nowcast input.

STATUS: FRAMEWORK ONLY - NO DATA SOURCE WIRED YET. Every function returns
empty/raises rather than inventing numbers (PLAN_1.md rules).

Why it matters: ~2.5% base weight but +-10-20% monthly swings; single months
have contributed +-0.3-0.5pp to the headline (PLAN_1.md Phase 3).

Design (to implement when a collection path is agreed):
- Hagstofa prices international airfares from fares OBSERVED DURING THE
  COLLECTION WINDOW (first ~15 days of month t) for departures at fixed
  booking horizons. A scraper must therefore quote, each day d in the window,
  a fixed basket: routes x departure-offsets x fare class.
- Routes: KEF to the main Icelandair/PLAY destinations weighted by seat
  capacity (CPH, OSL, ARN, LHR, AMS, CDG, BER, ALC, TFS, BOS, JFK, ...).
- Offsets: departures at ~2, 6, 10 weeks after quote date (matching the mix of
  booking horizons in Hagstofa's spec; calibrate the mix on the published
  subindex).
- Store raw quotes to data/airfares/quotes_YYYY-MM-DD.csv; monthly index =
  mean log fare per (route, offset) cell averaged over the window, Jevons
  across cells; calibrate cell weights against published CP07332 m/m.
- Collection options: (a) headless browser against booking flows (fragile,
  ToS review needed), (b) a fares API (Amadeus/Kiwi/Dohop) with a key, or
  (c) manual weekly quote entry into the CSV (low-tech but unbiased).

Interface mirrors fuel.py so nowcast.py can consume it unchanged.
"""
from __future__ import annotations

import pandas as pd

from .px_client import REPO_ROOT

QUOTE_DIR = REPO_ROOT / "data" / "airfares"

QUOTE_COLUMNS = [
    "quote_date",      # date the fare was observed (must lie in collection window)
    "carrier",         # ICE / PLAY / other
    "origin", "dest",  # IATA codes, origin normally KEF
    "depart_date",     # departure date of the quoted itinerary
    "fare_isk",        # total price incl. taxes, one adult, lowest available
    "fare_class",      # e.g. Economy Light
    "source",          # scraper id or 'manual'
]


def load_quotes() -> pd.DataFrame:
    """All stored fare quotes; empty frame (correct columns) when none exist."""
    if not QUOTE_DIR.exists():
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    frames = [pd.read_csv(f, parse_dates=["quote_date", "depart_date"])
              for f in sorted(QUOTE_DIR.glob("quotes_*.csv"))]
    if not frames:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def airfare_index_mm() -> pd.Series:
    """Matched-cell Jevons m/m of collection-window fares. Empty until quotes exist."""
    q = load_quotes()
    if q.empty:
        return pd.Series(dtype=float, name="airfare_mm")
    q = q[q.quote_date.dt.day <= 15].copy()
    q["manudur"] = q.quote_date.dt.to_period("M")
    q["offset_w"] = ((q.depart_date - q.quote_date).dt.days // 28).clip(0, 3)
    import numpy as np
    cell = q.groupby(["carrier", "origin", "dest", "offset_w", "manudur"]).fare_isk.apply(
        lambda s: np.log(s).mean())
    rel = cell.groupby(level=[0, 1, 2, 3]).diff()
    mm = rel.groupby(level="manudur").mean()
    return ((np.exp(mm) - 1) * 100).rename("airfare_mm")
