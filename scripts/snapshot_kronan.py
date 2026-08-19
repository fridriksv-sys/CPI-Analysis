"""OPTION 2 for grocery price history: local daily snapshot via api.kronan.is.

Requires KRONAN_API_TOKEN in the environment (same token home_app's Supabase
functions use). Walks every leaf category and writes
data/kronan/snapshot_YYYY-MM-DD.csv. Run daily (Task Scheduler / cron) during
at least days 1-15 of each month to cover Hagstofa's collection window.

Usage:  set KRONAN_API_TOKEN=...   (or $env:KRONAN_API_TOKEN='...')
        .venv\\Scripts\\python.exe scripts\\snapshot_kronan.py
"""
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "kronan"
BASE = "https://api.kronan.is/api/v1"
TOKEN = os.environ.get("KRONAN_API_TOKEN")

# Same rate-limit discipline as home_app's kronan.ts: 200 req / 200 s shared.
PAUSE_S = 1.1
MAX_PAGES_PER_LEAF = 6


def get(url: str, retries: int = 2):
    for attempt in range(retries + 1):
        r = requests.get(url, headers={"Authorization": f"AccessToken {TOKEN}"}, timeout=60)
        if r.ok:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed after retries: {url}")


def leaf_slugs() -> list[str]:
    def collect(node, acc):
        kids = node.get("children") or []
        if not kids:
            acc.append(node["slug"])
        for c in kids:
            collect(c, acc)
    acc: list[str] = []
    for top in get(f"{BASE}/categories/"):
        collect(top, acc)
    return acc


def main() -> int:
    if not TOKEN:
        print("KRONAN_API_TOKEN not set - refusing to run (no synthetic data).")
        return 1
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAP_DIR / f"snapshot_{date.today().isoformat()}.csv"
    if out_path.exists():
        print(f"{out_path.name} already exists - one snapshot per day.")
        return 0

    rows = []
    slugs = leaf_slugs()
    print(f"{len(slugs)} leaf categories")
    for i, slug in enumerate(slugs):
        for page in range(1, MAX_PAGES_PER_LEAF + 1):
            data = get(f"{BASE}/categories/{slug}/products/?page={page}")
            for p in data.get("products", []):
                rows.append({
                    "sku": p.get("sku"), "name": p.get("name"),
                    "price": p.get("price"), "discounted_price": p.get("discountedPrice"),
                    "on_sale": bool(p.get("onSale")), "unit": p.get("baseComparisonUnit"),
                    "category_path": p.get("categoryPath"),
                })
            time.sleep(PAUSE_S)
            if not data.get("hasNextPage"):
                break
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(slugs)} leaves, {len(rows)} rows")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
