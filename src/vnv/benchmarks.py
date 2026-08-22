"""External benchmarks: breakeven inflation, bank analysts, Seðlabanki (Phase 7).

These are not cleanly machine-accessible:
- Breakeven inflation (RIKS vs RIKB) is the tradeable benchmark and the whole
  economic point of the model (the h=3-12 edge). The clean source is a market
  data feed (LSEG connector — needs auth; or Keldan/Kodiak). Seðlabanki publishes
  verðbólguálag in monetary-policy reports but not via the open data API.
- Bank analyst forecasts (Íslandsbanki Greining, Landsbankinn, Arion) and
  Seðlabanki Peningamál are published in notes/PDFs ~monthly/quarterly.

So both are read from committed CSV slots the user (or a future feed) populates.
The comparison code is ready; only the data is pending. NO synthetic values —
missing files return empty frames.
"""
from __future__ import annotations

import pandas as pd

from .px_client import REPO_ROOT

DATA = REPO_ROOT / "data" / "benchmarks"

BREAKEVEN_COLS = ["date", "horizon_yrs", "breakeven_pct"]
ANALYST_COLS = ["forecast_date", "target_month", "source", "mm_pct", "yoy_pct"]


def load_breakeven() -> pd.DataFrame:
    """Breakeven inflation curve, tidy (date, horizon_yrs, breakeven_pct).

    Populate data/benchmarks/breakeven.csv from the market feed. Empty until then.
    """
    f = DATA / "breakeven.csv"
    if not f.exists():
        return pd.DataFrame(columns=BREAKEVEN_COLS)
    df = pd.read_csv(f, parse_dates=["date"])
    return df[BREAKEVEN_COLS]


def load_analyst_forecasts() -> pd.DataFrame:
    """Bank-analyst + Seðlabanki forecasts, tidy. Populate
    data/benchmarks/analyst_forecasts.csv. Empty until then."""
    f = DATA / "analyst_forecasts.csv"
    if not f.exists():
        return pd.DataFrame(columns=ANALYST_COLS)
    return pd.read_csv(f, parse_dates=["forecast_date"])


def model_vs_breakeven(model_yoy_12m: float, horizon_yrs: float = 1.0):
    """Compare the model's near-term inflation to the shortest market breakeven.

    Breakeven = market expected inflation + inflation-risk premium + indexed-bond
    scarcity premium. The model's clean output is 12-month inflation, while the
    shortest indexed bond (RIKS) matures ~4y out, so this compares the model's
    near-term call to the SHORTEST available breakeven and flags the horizon gap;
    the difference (breakeven − model) is an indicative premium/expectations wedge,
    not an exact decomposition. Returns None until the breakeven slot is populated.
    """
    be = load_breakeven()
    if be.empty:
        return None
    latest = be[be.date == be.date.max()].sort_values("horizon_yrs")
    row = latest.iloc[0]  # shortest available horizon (nearest the model's horizon)
    breakeven = float(row.breakeven_pct)
    return {"model_yoy": model_yoy_12m, "breakeven": breakeven,
            "implied_wedge": breakeven - model_yoy_12m,
            "breakeven_horizon_yrs": float(row.horizon_yrs),
            "curve": latest[["horizon_yrs", "breakeven_pct"]].to_dict("records")}


def write_templates():
    """Create empty, correctly-headed CSV slots (does not overwrite existing)."""
    DATA.mkdir(parents=True, exist_ok=True)
    for name, cols in [("breakeven.csv", BREAKEVEN_COLS),
                       ("analyst_forecasts.csv", ANALYST_COLS)]:
        p = DATA / name
        if not p.exists():
            pd.DataFrame(columns=cols).to_csv(p, index=False)
    return str(DATA)
