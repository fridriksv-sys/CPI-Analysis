"""Seðlabanki Íslands FX data: the ISK exchange-rate index (gengisvísitala).

The gengisvísitala (series 4118) is Iceland's nominal effective exchange rate
index. Convention: a HIGHER index = a WEAKER króna (depreciation), so imported-
goods CPI moves *with* the index. It's the standard NEER proxy for pass-through.

Access: Seðlabanki's official-rate CSV API returns the registered index value
as of any date:
  https://sedlabanki.is/api/rate/csv?timeseries=4118&date=YYYY-MM-DD
We sample the value around day 15 of each month — the middle of Hagstofa's price-
collection window — so the monthly FX lines up with the prices it feeds. Results
are cached to data/raw/. (The interactive time-series endpoint is Blazor-WASM
with obfuscated params; this dated-snapshot API is the stable programmatic path.)
"""
from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
import requests

from pathlib import Path

from .px_client import RAW_DIR, REPO_ROOT

API = "https://sedlabanki.is/api/rate/csv"
GENGISVISITALA = "4118"
START = date(2020, 1, 1)  # earliest available in the source
_UA = {"User-Agent": "Mozilla/5.0"}


def _value_asof(d: date) -> float | None:
    r = requests.get(API, params={"timeseries": GENGISVISITALA, "date": d.isoformat(),
                                  "showDates": "True"}, timeout=30, headers=_UA)
    if r.status_code != 200:
        return None
    txt = r.content.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(txt), sep=";")
    except Exception:
        return None
    col = next((c for c in df.columns if "krán" in c.lower() or "skrán" in c.lower()), None)
    if col is None or df.empty or df[col].isna().all():
        return None
    val = str(df[col].dropna().iloc[0]).replace(".", "").replace(",", ".")
    try:
        return float(val)
    except ValueError:
        return None


def _midmonth_value(year: int, month: int) -> float | None:
    """Index as of ~day 15; step outward to the nearest registered (non-holiday) day."""
    for day in (15, 16, 14, 17, 13, 18, 12, 19, 11, 20, 10):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        v = _value_asof(d)
        if v is not None:
            return v
    return None


# Committed seed so a fresh deploy (e.g. Streamlit Cloud) starts instantly instead
# of making ~80 sequential API calls. Refreshed by save_fx_seed() and committed.
SEED = REPO_ROOT / "data" / "fx" / "gengisvisitala_monthly.csv"


def _read_fx_csv(path: Path) -> pd.Series:
    s = pd.read_csv(path)
    return pd.Series(s.gengisvisitala.values,
                     index=pd.PeriodIndex(s.manudur, freq="M"), name="gengisvisitala")


def load_fx(use_cache: bool = True) -> pd.Series:
    """Monthly ISK gengisvísitala (mid-collection-window sample), indexed by month.

    Order: live cache (fresh, local) -> committed seed (instant, cloud) -> fetch.
    """
    cache = RAW_DIR / "sedlabanki_gengisvisitala_monthly.csv"
    if use_cache and cache.exists():
        return _read_fx_csv(cache)
    if use_cache and SEED.exists():
        return _read_fx_csv(SEED)

    today = date.today()
    rows = []
    y, m = START.year, START.month
    while (y, m) <= (today.year, today.month):
        v = _midmonth_value(y, m)
        if v is not None:
            rows.append((f"{y}-{m:02d}", v))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    df = pd.DataFrame(rows, columns=["manudur", "gengisvisitala"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    return pd.Series(df.gengisvisitala.values,
                     index=pd.PeriodIndex(df.manudur, freq="M"), name="gengisvisitala")


def fx_mm(use_cache: bool = True) -> pd.Series:
    """m/m % change of the ISK NEER (positive = depreciation)."""
    return (load_fx(use_cache=use_cache).pct_change() * 100).rename("fx_mm")


def save_fx_seed(use_cache: bool = False) -> str:
    """Fetch the monthly FX series and write the committed seed (data/fx/).

    Run before committing so a fresh deploy starts instantly:
      .venv\\Scripts\\python.exe -c "from vnv import sedlabanki; sedlabanki.save_fx_seed()"
    """
    s = load_fx(use_cache=use_cache)
    SEED.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"manudur": s.index.astype(str), "gengisvisitala": s.values}).to_csv(
        SEED, index=False, encoding="utf-8")
    return str(SEED)
