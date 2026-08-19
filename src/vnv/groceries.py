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
    """All local snapshots as one frame (snapshot_date, sku, price, category_path)."""
    if not SNAP_DIR.exists():
        return pd.DataFrame(columns=["snapshot_date", "sku", "price", "category_path"])
    frames = []
    for f in sorted(SNAP_DIR.glob("snapshot_*.csv")):
        df = pd.read_csv(f, encoding="utf-8")
        df["snapshot_date"] = pd.to_datetime(f.stem.replace("snapshot_", ""))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["snapshot_date", "sku", "price", "category_path"])
    return pd.concat(frames, ignore_index=True)


def _coicop_class(category_path: str) -> str | None:
    top = str(category_path).split("/")[0].strip()
    return CATEGORY_TO_COICOP.get(top)


def food_index_mm(collection_day_to: int = 15) -> pd.DataFrame:
    """Matched-model m/m per COICOP class from the snapshot history.

    Within class: Jevons (geometric mean of price relatives of SKUs present in
    both months' collection windows). Month price = mean over snapshots with
    day <= collection_day_to (Hagstofa collection window).
    Returns empty frame until enough snapshots exist - never synthesizes.
    """
    snaps = load_snapshots()
    if snaps.empty:
        return pd.DataFrame()
    snaps = snaps.dropna(subset=["price"])
    snaps["coicop"] = snaps["category_path"].map(_coicop_class)
    snaps = snaps.dropna(subset=["coicop"])
    snaps = snaps[snaps.snapshot_date.dt.day <= collection_day_to]
    snaps["manudur"] = snaps.snapshot_date.dt.to_period("M")

    monthly = (
        snaps.groupby(["coicop", "sku", "manudur"], as_index=False)
        .agg(price=("price", "mean"))
    )
    out = []
    for (coicop, sku), grp in monthly.groupby(["coicop", "sku"]):
        s = grp.set_index("manudur").price.sort_index()
        rel = np.log(s / s.shift(1))
        for m, v in rel.dropna().items():
            out.append({"coicop": coicop, "sku": sku, "manudur": m, "log_rel": v})
    if not out:
        return pd.DataFrame()
    rels = pd.DataFrame(out)
    idx = (
        rels.groupby(["coicop", "manudur"])
        .agg(mm=("log_rel", lambda v: (np.exp(v.mean()) - 1) * 100), n_matched=("sku", "count"))
        .reset_index()
    )
    return idx
