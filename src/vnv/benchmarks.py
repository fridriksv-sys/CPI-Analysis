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

import numpy as np
import pandas as pd

from .px_client import REPO_ROOT

DATA = REPO_ROOT / "data" / "benchmarks"

BREAKEVEN_COLS = ["date", "horizon_yrs", "breakeven_pct"]
ANALYST_COLS = ["forecast_date", "target_month", "source", "mm_pct", "yoy_pct"]
PENINGAMAL_COLS = ["forecast_date", "target_quarter", "yoy_pct"]
ANALYST_ANNUAL_COLS = ["forecast_date", "source", "year", "yoy_pct"]


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
    data/benchmarks/analyst_forecasts.csv (one row per source per target month).
    Empty until then. Real published figures only — never synthesize."""
    f = DATA / "analyst_forecasts.csv"
    if not f.exists():
        return pd.DataFrame(columns=ANALYST_COLS)
    df = pd.read_csv(f, parse_dates=["forecast_date"])
    df["manudur"] = pd.PeriodIndex(df.target_month.astype(str), freq="M")
    return df


def load_analyst_annual() -> pd.DataFrame:
    """Bank analysts' ANNUAL (calendar-year) inflation forecasts, from their
    semi-annual macro forecasts (Íslandsbanki þjóðhagsspá, Landsbankinn hagspá).
    Populate data/benchmarks/analyst_annual.csv. Real published figures only."""
    f = DATA / "analyst_annual.csv"
    if not f.exists():
        return pd.DataFrame(columns=ANALYST_ANNUAL_COLS)
    return pd.read_csv(f, parse_dates=["forecast_date"])


def analyst_term_structure(base_year: int) -> list[dict] | None:
    """Named analysts' annual forecasts mapped to years-ahead (year − base_year),
    latest vintage per source, for overlay on the term-structure comparison.
    Calendar year Y ≈ (Y − base_year) years ahead (2027 ≈ 1yr from mid-2026)."""
    aa = load_analyst_annual()
    if aa.empty:
        return None
    out = []
    for src, d in aa.groupby("source"):
        d = d[d.forecast_date == d.forecast_date.max()]
        for r in d.itertuples():
            h = int(r.year) - base_year
            if h >= 1:  # only forward years (1yr, 2yr, 3yr ahead)
                out.append({"source": src, "horizon_yrs": h, "yoy_pct": float(r.yoy_pct),
                            "year": int(r.year), "forecast_date": r.forecast_date})
    return out or None


def load_peningamal() -> pd.DataFrame:
    """Seðlabanki Peningamál quarterly y/y inflation forecast (Tafla 5).

    Populate data/benchmarks/peningamal.csv from each quarterly Peningamál
    (published ~Feb/May/Aug/Nov). Real published figures only."""
    f = DATA / "peningamal.csv"
    if not f.exists():
        return pd.DataFrame(columns=PENINGAMAL_COLS)
    df = pd.read_csv(f, parse_dates=["forecast_date"])
    df["quarter"] = pd.PeriodIndex(df.target_quarter.str.replace("Q", "Q"), freq="Q")
    return df


def peningamal_comparison(yy_path_monthly: pd.Series) -> dict | None:
    """Model's quarterly-average y/y vs the latest Peningamál forecast, over the
    quarters the model path covers. yy_path_monthly: reconciled y/y by month."""
    pm = load_peningamal()
    if pm.empty or yy_path_monthly.empty:
        return None
    pm = pm[pm.forecast_date == pm.forecast_date.max()]
    model_q = yy_path_monthly.groupby(yy_path_monthly.index.asfreq("Q")).mean()
    rows = []
    for r in pm.itertuples():
        if r.quarter in model_q.index:
            rows.append({"quarter": str(r.quarter), "model_yoy": round(float(model_q[r.quarter]), 2),
                         "peningamal_yoy": r.yoy_pct, "diff": round(float(model_q[r.quarter]) - r.yoy_pct, 2)})
    if not rows:
        return None
    return {"vintage": pm.forecast_date.max(), "rows": rows,
            "full_curve": pm[["target_quarter", "yoy_pct"]].to_dict("records")}


def analyst_comparison(head, model_nowcast: float, nowcast_month) -> dict | None:
    """Model vs analysts vs consensus for the upcoming print, plus the realized
    track record (each source's m/m error vs the published actual).

    head: VIS01000 frame (for the published actual m/m). Returns None if no
    analyst rows exist.
    """
    fc = load_analyst_forecasts()
    if fc.empty:
        return None
    actual_mm = head[("CPI", "change_M")]

    # --- upcoming print: model vs each analyst vs consensus ---
    up = fc[fc.manudur == nowcast_month]
    current = None
    if not up.empty:
        rows = [{"source": r.source, "mm_pct": r.mm_pct, "yoy_pct": r.yoy_pct,
                 "made_on": r.forecast_date} for r in up.itertuples()]
        cons = float(up.mm_pct.mean())
        current = {"month": nowcast_month, "model": model_nowcast,
                   "analysts": rows, "consensus_mm": cons,
                   "model_minus_consensus": model_nowcast - cons}

    # --- realized track record: analyst m/m error vs published actual ---
    real = fc[fc.manudur.isin(actual_mm.index)].copy()
    real["actual"] = real.manudur.map(actual_mm)
    real = real.dropna(subset=["actual"])
    track = None
    if not real.empty:
        real["error"] = real.mm_pct - real.actual
        track = (real.groupby("source")
                 .apply(lambda d: pd.Series({
                     "n": len(d),
                     "RMSE": float(np.sqrt((d.error ** 2).mean())),
                     "MAE": float(d.error.abs().mean()),
                     "bias": float(d.error.mean())}), include_groups=False)
                 .reset_index())
        detail = real[["manudur", "source", "mm_pct", "actual", "error"]].sort_values("manudur")
    else:
        detail = None
    return {"current": current, "track": track, "detail": detail}


def model_vs_breakeven(model_term: dict):
    """Compare the model's inflation TERM STRUCTURE to the market breakeven curve.

    A T-year breakeven ≈ the market's average annual inflation over [now, now+T]
    (nominal RIKB − real RIKS yield), so the horizon-matched comparison is the
    model's own average annual inflation over the same horizon. `model_term` maps
    {years: avg annual inflation %} (e.g. from the 36-month path, 1/2/3 years).

    The wedge is taken at the shortest breakeven (~4y) matched to the model's
    longest term (3y) — the closest the horizons align. Breakeven above the model
    is the inflation-risk + indexed-bond-scarcity premium (where the h=3–12 edge
    lives); it is an indicative wedge, not an exact decomposition. Returns None
    until the breakeven slot is populated.
    """
    be = load_breakeven()
    if be.empty or not model_term:
        return None
    be_date = be.date.max()
    latest = be[be.date == be_date].sort_values("horizon_yrs")
    row = latest.iloc[0]                       # shortest breakeven (~4y)
    be_h = float(row.horizon_yrs)
    myr = min(model_term, key=lambda y: abs(y - be_h))   # nearest model term (3y)
    model_avg = float(model_term[myr])
    breakeven = float(row.breakeven_pct)
    return {"model_avg": model_avg, "model_horizon_yrs": float(myr),
            "breakeven": breakeven, "breakeven_horizon_yrs": be_h,
            "breakeven_date": be_date,
            "implied_wedge": breakeven - model_avg,
            "model_term": [{"horizon_yrs": float(y), "avg_infl": float(v)}
                           for y, v in sorted(model_term.items())],
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
