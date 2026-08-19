"""Groceries: Kronan price snapshots -> matched-model food index (CP011x).

Reuses the existing Kronan catalog mirror built for home_app (Supabase table
`grocery_catalog`, synced from api.kronan.is). The mirror holds CURRENT prices
only, so a history has to accumulate as daily snapshots. Two supported feeds:

1. (preferred) Server-side: `kronan_price_history` table + pg_cron job in the
   home_app Supabase project appending a daily snapshot; export or query it
   into data/kronan/*.csv. Migration SQL: scripts/kronan_history_migration.sql
   - NOT applied; requires owner approval.
2. Local: scripts/snapshot_kronan.py pulls the catalog via api.kronan.is
   directly (requires KRONAN_API_TOKEN in the environment) and writes
   data/kronan/snapshot_YYYY-MM-DD.csv.

Once >= 2 snapshots straddle a month boundary, `food_index_mm()` produces a
matched-model (Jevons within class, Laspeyres across classes) m/m estimate to
calibrate against CP011x - the same design as the fuel leg.

NO SYNTHETIC DATA: every function raises/returns empty when snapshots are
missing rather than inventing prices (PLAN_1.md rules).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .px_client import REPO_ROOT

SNAP_DIR = REPO_ROOT / "data" / "kronan"

# Kronan top-level category -> COICOP2018 class mapping (food and beverages).
# Non-food categories (Snyrtivara, Heimilid, Dyrin, ...) are intentionally
# excluded from the food index; they may later feed CP05/CP13 checks.
CATEGORY_TO_COICOP = {
    "Brauð, kökur og kex": "CP0111",
    "Kjöt": "CP0112",
    "Fiskur": "CP0113",
    "Mjólkurvörur og egg": "CP0114",
    "Ávextir": "CP0116",
    "Grænmeti": "CP0117",
    "Laugardags": "CP0118",        # saelgaeti -> sugar/confectionery
    "Eldamennskan": "CP0119",      # cooking staples -> other food n.e.c.
    "Morgunmatur og heilsubót": "CP0111",  # cereals
    "Á brauð": "CP0115",           # spreads -> oils/fats (approx)
    "Bakstur": "CP0119",
    "Tilbúnir réttir": "CP0119",
    "Frystivara": "CP0119",
    "Drykkir": "CP012",            # non-alcoholic beverages
    "Heitir drykkir": "CP0121",
}


def load_snapshots() -> pd.DataFrame:
    """Local price history as one frame (snapshot_date, sku, price, category_path).

    Reads data/kronan/history.csv (export of the server-side kronan_price_history
    change-log; see scripts/export_kronan_history.py) plus any snapshot_*.csv
    files from the local scraper. Change-log semantics: a (date, sku) row means
    the price CHANGED that day; prices between rows are reconstructed by
    forward-fill in daily_prices().
    """
    frames = []
    hist = SNAP_DIR / "history.csv"
    if hist.exists():
        df = pd.read_csv(hist, encoding="utf-8", parse_dates=["snapshot_date"])
        frames.append(df)
    if SNAP_DIR.exists():
        for f in sorted(SNAP_DIR.glob("snapshot_*.csv")):
            df = pd.read_csv(f, encoding="utf-8")
            df["snapshot_date"] = pd.to_datetime(f.stem.replace("snapshot_", ""))
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["snapshot_date", "sku", "price", "category_path"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["snapshot_date", "sku"])


def daily_prices(snaps: pd.DataFrame) -> pd.DataFrame:
    """sku x day price matrix, forward-filled between stored change rows."""
    piv = snaps.pivot_table(index="snapshot_date", columns="sku", values="price", aggfunc="last")
    full = pd.date_range(piv.index.min(), pd.Timestamp.today().normalize(), freq="D")
    return piv.reindex(full).ffill()


def _coicop_class(category_path: str) -> str | None:
    top = str(category_path).split("/")[0].strip()
    return CATEGORY_TO_COICOP.get(top)


def food_index_mm(collection_day_to: int = 15) -> pd.DataFrame:
    """Matched-model m/m per COICOP class from the price history.

    Month price per SKU = mean of the forward-filled daily price over the
    collection window (days 1..collection_day_to), matching Hagstofa's method.
    Within class: Jevons (geometric mean of price relatives of SKUs priced in
    both consecutive windows). Returns an empty frame until the history spans
    at least two collection windows - never synthesizes.
    """
    snaps = load_snapshots()
    if snaps.empty:
        return pd.DataFrame()
    snaps = snaps.dropna(subset=["price"])
    coicop = (
        snaps.sort_values("snapshot_date")
        .groupby("sku")["category_path"].last().map(_coicop_class).dropna()
    )
    daily = daily_prices(snaps)
    window = daily[daily.index.day <= collection_day_to]
    monthly = window.groupby(window.index.to_period("M")).mean()
    if len(monthly) < 2:
        return pd.DataFrame()

    log_rel = np.log(monthly / monthly.shift(1))
    out = []
    for m, row in log_rel.iloc[1:].iterrows():
        rel = row.dropna()
        rel = rel[rel.index.isin(coicop.index)]
        grp = rel.groupby(coicop.reindex(rel.index))
        for cls, vals in grp:
            out.append({"coicop": cls, "manudur": m,
                        "mm": (np.exp(vals.mean()) - 1) * 100, "n_matched": len(vals)})
    return pd.DataFrame(out)
