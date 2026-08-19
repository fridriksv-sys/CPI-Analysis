"""Airfares (CP0733 / CP07332): the highest-leverage nowcast input.

Collection path (the easiest robust one found): Icelandair's Icelandic-edition
destination pages server-render their lowest available fares into the page HTML
(`__NEXT_DATA__` -> apolloState -> StandardFareModule.fares). Each page yields
KEF-origin ISK fares for one destination market. We fetch a fixed route set,
record the lowest fare per (destination, one-way/round-trip) each day, and
calibrate the monthly collection-window index against the published CP07332.

Why this design (vs a live fare-search API):
- No API key, no auth, no CORS: a plain GET returns the fares baked into HTML.
- Cloudflare bot-protects these pages, so requests must present a real browser
  TLS fingerprint -> curl_cffi impersonation (see fetch_page).
- It is a lowest-available-fare feed, not a fixed-basket quote, so like the fuel
  leg it is used as a CALIBRATED PROXY: consistent daily collection of the same
  routes gives a signal that co-moves with the published index; the calibration
  regression absorbs the level/scale difference. NOT a replacement for Hagstofa.

NO SYNTHETIC DATA: on a failed fetch a route is skipped and logged, never
filled with an estimate (PLAN_1.md rules).
"""
from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd

from .px_client import REPO_ROOT

QUOTE_DIR = REPO_ROOT / "data" / "airfares"

# Fixed route set: KEF -> major Icelandair international markets, weighted toward
# Hagstofa's likely basket (Scandinavia + hubs + sun routes). Slugs are the
# Icelandic-edition destination pages.
ROUTES = {
    "CPH": "flug-til-kaupmannahafnar",
    "LON": "flug-til-london",
    "OSL": "flug-til-osloar",
    "ARN": "flug-til-stokkholms",
    "AMS": "flug-til-amsterdam",
    "CDG": "flug-til-parisar",
    "BER": "flug-til-berlinar",
    "ALC": "flug-til-alicante",
    "TFS": "flug-til-tenerife",
    "BCN": "flug-til-barcelona",
    "JFK": "flug-til-new-york",
    "BOS": "flug-til-boston",
}
BASE = "https://www.icelandair.com/is-is/flug/"

QUOTE_COLUMNS = [
    "quote_date", "carrier", "origin", "dest", "depart_date", "return_date",
    "flight_type", "fare_isk", "fare_class", "source",
]


def fetch_page(slug: str) -> str | None:
    """GET one destination page past Cloudflare via browser TLS impersonation."""
    from curl_cffi import requests as creq
    try:
        r = creq.get(BASE + slug, headers={"Accept-Language": "is-IS,is;q=0.9"},
                     impersonate="chrome124", timeout=60)
    except Exception:
        return None
    return r.text if r.status_code == 200 else None


def parse_fares(html: str) -> list[dict]:
    """Extract KEF-origin fares from a destination page's __NEXT_DATA__."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))["props"]["pageProps"]["apolloState"]["data"]
    except (KeyError, json.JSONDecodeError):
        return []
    out = []
    for key in (k for k in data if k.startswith("StandardFareModule") and data[k].get("fares")):
        for ref in data[key]["fares"]:
            f = data.get(ref["__ref"], ref) if isinstance(ref, dict) and "__ref" in ref else ref
            if not isinstance(f, dict):
                continue
            if f.get("originAirportCode") != "KEF" or f.get("currencyCode") != "ISK":
                continue
            out.append({
                "carrier": "ICE",
                "origin": "KEF",
                "dest": f.get("destinationAirportCode"),
                "depart_date": f.get("departureDate"),
                "return_date": f.get("returnDate"),
                "flight_type": f.get("flightType"),
                "fare_isk": f.get("totalPrice"),
                "fare_class": (f.get("travelClass") or "").upper(),
            })
    return out


def collect(routes: dict[str, str] | None = None) -> pd.DataFrame:
    """Fetch all routes and return today's lowest KEF-origin fare per (dest, type).

    Writes data/airfares/quotes_YYYY-MM-DD.csv (idempotent per day). Only the
    lowest fare per destination and flight-type is kept — the marketing feed
    lists several dates per route; the minimum is the consistent daily signal.
    """
    routes = routes or ROUTES
    today = date.today().isoformat()
    rows = []
    for dest, slug in routes.items():
        html = fetch_page(slug)
        if html is None:
            print(f"  skip {dest}: fetch failed")
            continue
        for f in parse_fares(html):
            if f["fare_isk"] is None or f["dest"] != dest:
                continue
            rows.append({**f, "quote_date": today, "source": "icelandair_is_ssr"})
    if not rows:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    df = pd.DataFrame(rows)
    lowest = (
        df.sort_values("fare_isk")
        .groupby(["dest", "flight_type"], as_index=False)
        .first()[QUOTE_COLUMNS]
    )
    QUOTE_DIR.mkdir(parents=True, exist_ok=True)
    out = QUOTE_DIR / f"quotes_{today}.csv"
    lowest.to_csv(out, index=False, encoding="utf-8")
    return lowest


def load_quotes() -> pd.DataFrame:
    """All stored fare quotes; empty frame (correct columns) when none exist."""
    if not QUOTE_DIR.exists():
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    frames = [pd.read_csv(f) for f in sorted(QUOTE_DIR.glob("quotes_*.csv"))]
    if not frames:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def airfare_index_mm(collection_day_to: int = 15) -> pd.Series:
    """Matched-route Jevons m/m of collection-window lowest fares.

    Month fare per route = mean of the daily lowest fare over days 1..15.
    Within month: geometric mean of route-level price relatives for routes
    priced in both consecutive windows. Empty until two windows exist.
    """
    q = load_quotes()
    if q.empty:
        return pd.Series(dtype=float, name="airfare_mm")
    q = q.dropna(subset=["fare_isk"]).copy()
    q["quote_date"] = pd.to_datetime(q["quote_date"])
    q = q[q.quote_date.dt.day <= collection_day_to]
    q["manudur"] = q.quote_date.dt.to_period("M")
    q["route"] = q.dest + "_" + q.flight_type.fillna("NA")

    monthly = q.groupby(["route", "manudur"]).fare_isk.mean().unstack("route")
    if len(monthly) < 2:
        return pd.Series(dtype=float, name="airfare_mm")
    log_rel = np.log(monthly / monthly.shift(1))
    return ((np.exp(log_rel.mean(axis=1)) - 1) * 100).dropna().rename("airfare_mm")
