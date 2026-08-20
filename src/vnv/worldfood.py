"""FAO Food Price Index — world food commodity prices (USD terms).

A driver for the domestic food CPI (CP01) alongside FX: Iceland imports much of
its food, so ISK-price of food ~ FAO index (USD) x ISK/USD, with a pass-through
lag. Source CSV updates monthly (nominal + real, 1990-).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .px_client import RAW_DIR

CSV_URL = ("https://www.fao.org/media/docs/worldfoodsituationlibraries/"
           "default-document-library/food_price_indices_data.csv?download=true")


def load_fao(use_cache: bool = True) -> pd.DataFrame:
    """FAO Food Price Index, monthly, indexed by month.

    Returns the headline 'Food Price Index' plus the commodity sub-indices
    (Meat, Dairy, Cereals, Oils, Sugar) where present.
    """
    cache = RAW_DIR / "fao_food_price_index.csv"
    if use_cache and cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        r = requests.get(CSV_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")

    # The file has a preamble; find the header row that starts with a Date column.
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if ln.lower().split(",")[0].strip('" ') in ("date", "mánuður", "month"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["manudur"] = df[date_col].dt.to_period("M")
    df = df.set_index("manudur").drop(columns=[date_col])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def food_index(use_cache: bool = True) -> pd.Series:
    """Headline FAO Food Price Index as a monthly series."""
    df = load_fao(use_cache=use_cache)
    col = next((c for c in df.columns if "food price index" in c.lower()),
               next((c for c in df.columns if "food" in c.lower()), df.columns[0]))
    return df[col].rename("fao_food")


def food_mm(use_cache: bool = True) -> pd.Series:
    """m/m % change of the FAO Food Price Index (USD terms)."""
    return (food_index(use_cache=use_cache).pct_change() * 100).rename("fao_food_mm")
