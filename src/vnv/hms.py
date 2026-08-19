"""HMS housing data: leiguvísitala (rent) and kaupvísitala (house price).

Source: HMS (Húsnæðis-, mannvirkja- og skipulagsstofnun) publishes both indices
as open CSVs on OCI object storage, linked from
https://hms.is/gogn-og-maelabord/visitolur.

Series facts (from the HMS page):
- Rental index (leiguvísitala): the combined series is 100 in May 2023; the
  older capital-area rent index ran Jan 2011–May 2023. New-contract based,
  capital-area-weighted. This is the DRIVER for Hagstofa's stock-based
  reiknuð húsaleiga (CP042).
- Purchase-price index (kaupvísitala): new quality-adjusted index effective
  Jan 2024, back-calculated to Jan 2020; regional + house-type subindices.

Vintage handling: the CSV carries a per-row UTGAFUDAGUR (first-publication date;
populated for recent months in kaupvisitala, uniformly re-stamped in
leiguvisitala). snapshot() appends the current pull to data/hms/history/ so a
true input-vintage panel accrues going forward — the backtest can then use the
value known at each forecast date rather than the final revised one.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

from .px_client import RAW_DIR, REPO_ROOT

BASE = ("https://frs3o1zldvgn.objectstorage.eu-frankfurt-1.oci.customer-oci.com"
        "/n/frs3o1zldvgn/b/public_data_for_download/o/")
FILES = {"leiga": "leiguvisitala.csv", "kaup": "kaupvisitala.csv"}
HIST_DIR = REPO_ROOT / "data" / "hms" / "history"

# HMS publishes the previous month's value on the first Wednesday of a month;
# a value for month t is therefore knowable from ~t+1. Used by the vintage-aware
# backtest as the publication lag when a true snapshot vintage is unavailable.
PUBLICATION_LAG_MONTHS = 1


def _fetch_csv(key: str, use_cache: bool = True) -> pd.DataFrame:
    cache = RAW_DIR / f"hms_{key}.csv"
    if use_cache and cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        r = requests.get(BASE + FILES[key], timeout=60)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(text))
    df["manudur"] = pd.PeriodIndex(
        df.AR.astype(int).astype(str) + "-" + df.MANUDUR.astype(int).astype(str).str.zfill(2),
        freq="M",
    )
    df["utgafudagur"] = pd.to_datetime(df.UTGAFUDAGUR.replace(" ", None), errors="coerce")
    return df


def load_rent(use_cache: bool = True) -> pd.DataFrame:
    """Combined HMS rental index (May 2023 = 100), indexed by month."""
    df = _fetch_csv("leiga", use_cache=use_cache)
    return df.set_index("manudur")[["VISITALA", "utgafudagur"]].rename(
        columns={"VISITALA": "leiguvisitala"}).sort_index()


def load_house(use_cache: bool = True) -> pd.DataFrame:
    """HMS purchase-price index with regional / house-type subindices."""
    df = _fetch_csv("kaup", use_cache=use_cache).set_index("manudur").sort_index()
    ren = {
        "VISITALA": "kaup_alls",
        "VISITALA_HOFUDBORGARSVAEDI": "kaup_hofudborg",
        "VISITALA_LANDSBYGGD": "kaup_landsbyggd",
        "VISITALA_SERBYLI_HOFUDBORGARSVAEDI": "kaup_serbyli_hb",
        "VISITALA_FJOLBYLI_HOFUDBORGARSVAEDI": "kaup_fjolbyli_hb",
        "VISITALA_SERBYLI_LANDSBYGGD": "kaup_serbyli_lb",
        "VISITALA_FJOLBYLI_LANDSBYGGD": "kaup_fjolbyli_lb",
    }
    cols = [c for c in ren if c in df.columns]
    return df[cols + ["utgafudagur"]].rename(columns=ren)


def snapshot() -> dict[str, str]:
    """Append today's raw pull to data/hms/history/ to accrue input vintages."""
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    written = {}
    for key, fname in FILES.items():
        r = requests.get(BASE + fname, timeout=60)
        r.raise_for_status()
        out = HIST_DIR / f"{key}_{today}.csv"
        out.write_bytes(r.content)
        written[key] = str(out)
    return written
