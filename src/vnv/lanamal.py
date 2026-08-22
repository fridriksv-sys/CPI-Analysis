"""Breakeven inflation from Icelandic government bonds (Lánamál ríkisins).

Breakeven = nominal yield (RIKB, óverðtryggð) − real yield (RIKS, verðtryggð) at
matched maturities. It is the market's inflation compensation and the tradeable
benchmark the model is measured against (PLAN_1.md §7); above the true
expectation it carries an inflation-risk premium and an indexed-bond scarcity
premium, which is where the model's h=3–12 edge lives.

Source: lanamal.is exposes a clean JSON endpoint per bond:
  /api/market/LoadChartData?orderbookId=<ID>&lang=is&from=YYYY-MM-DD&to=...&data=Yield
returning UTF-16LE {"chartData": [[date, yield], ...]} of end-of-day yields.
Server-side accessible (no auth); cached to data/raw/.

Note on maturities: the shortest indexed bond (RIKS) matures ~3.6y out, so the
shortest computable breakeven is ~3–4y. Short indexed bonds are scarce — that
scarcity is itself part of the premium the model helps decompose.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import requests

from .px_client import RAW_DIR

_API = "https://lanamal.is/api/market/LoadChartData"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# On-the-run series (orderbookId -> maturity). Names are PREFIX YY MMDD.
NOMINAL = {  # RIKB — óverðtryggð (nominal)
    "RIKB_26_1015": "2026-10-15", "RIKB_27_0415": "2027-04-15",
    "RIKB_28_1115": "2028-11-15", "RIKB_29_0416": "2029-04-16",
    "RIKB_31_0124": "2031-01-24", "RIKB_35_0917": "2035-09-17",
    "RIKB_38_0215": "2038-02-15", "RIKB_42_0217": "2042-02-17",
}
INDEXED = {  # RIKS — verðtryggð (real)
    "RIKS_29_0917": "2029-09-17", "RIKS_30_0701": "2030-07-01",
    "RIKS_33_0321": "2033-03-21", "RIKS_37_0115": "2037-01-15",
    "RIKS_50_0915": "2050-09-15",
}


def fetch_yield(orderbook_id: str, use_cache: bool = True) -> pd.Series:
    """Daily end-of-day yield (%) for one bond, indexed by date."""
    cache = RAW_DIR / f"lanamal_{orderbook_id}.json"
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        url = f"{_API}?orderbookId={orderbook_id}&lang=is&from=2015-01-01&to=2035-01-01&data=Yield"
        r = requests.get(url, headers=_UA, timeout=60)
        r.raise_for_status()
        data = json.loads(r.content.decode("utf-16-le", errors="replace").lstrip("﻿"))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    cd = data.get("chartData", [])
    if not cd:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([x[0] for x in cd])
    return pd.Series([x[1] for x in cd], index=idx, name=orderbook_id).sort_index()


def _yields_asof(bonds: dict, asof: pd.Timestamp, use_cache: bool) -> pd.DataFrame:
    """Years-to-maturity and yield for each bond as of `asof` (last obs <= asof)."""
    rows = []
    for ob, mat in bonds.items():
        s = fetch_yield(ob, use_cache=use_cache)
        s = s[s.index <= asof]
        if s.empty:
            continue
        ttm = (pd.Timestamp(mat) - asof).days / 365.25
        if ttm > 0:
            rows.append({"bond": ob, "ttm": ttm, "yield": float(s.iloc[-1])})
    return pd.DataFrame(rows).sort_values("ttm")


def breakeven_curve(asof: pd.Timestamp | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Breakeven = interp(nominal) − interp(real) at the overlapping maturities."""
    asof = pd.Timestamp(asof or date.today())
    nom = _yields_asof(NOMINAL, asof, use_cache)
    real = _yields_asof(INDEXED, asof, use_cache)
    if nom.empty or real.empty:
        return pd.DataFrame(columns=["horizon_yrs", "nominal", "real", "breakeven"])
    lo = max(nom.ttm.min(), real.ttm.min())
    hi = min(nom.ttm.max(), real.ttm.max())
    hs = [h for h in (3, 4, 5, 7, 10, 15) if lo <= h <= hi]
    if not hs:
        hs = [round((lo + hi) / 2, 1)]
    out = []
    for h in hs:
        n = np.interp(h, nom.ttm, nom["yield"])
        rr = np.interp(h, real.ttm, real["yield"])
        out.append({"horizon_yrs": float(h), "nominal": round(n, 3),
                    "real": round(rr, 3), "breakeven": round(n - rr, 3)})
    return pd.DataFrame(out)


def update_breakeven_slot(use_cache: bool = False) -> str:
    """Write the latest breakeven curve to data/benchmarks/breakeven.csv (the slot
    benchmarks.load_breakeven / report / model_vs_breakeven read)."""
    from .benchmarks import DATA
    cur = breakeven_curve(use_cache=use_cache)
    DATA.mkdir(parents=True, exist_ok=True)
    out = (cur.assign(date=pd.Timestamp(date.today()).normalize())
           .rename(columns={"breakeven": "breakeven_pct"})[["date", "horizon_yrs", "breakeven_pct"]])
    path = DATA / "breakeven.csv"
    out.to_csv(path, index=False)
    return str(path)
