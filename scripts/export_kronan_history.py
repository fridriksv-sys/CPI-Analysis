"""Export the server-side kronan_price_history change-log to data/kronan/history.csv.

Uses the Supabase PostgREST endpoint with the service-role key (paginated).
Set env vars:  SUPABASE_URL=https://vamspsjnwpfpocdiquzw.supabase.co
               SUPABASE_SERVICE_ROLE_KEY=...
(An in-session alternative: ask Claude to export via the Supabase MCP.)
"""
import csv
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kronan" / "history.csv"
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
PAGE = 10_000
COLS = ["snapshot_date", "sku", "price", "discounted_price", "on_sale", "unit", "category_path"]


def main() -> int:
    if not URL or not KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set - refusing to run.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    headers = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
    rows: list[dict] = []
    offset = 0
    while True:
        r = requests.get(
            f"{URL}/rest/v1/kronan_price_history",
            headers={**headers, "Range": f"{offset}-{offset + PAGE - 1}"},
            params={"select": ",".join(COLS), "order": "snapshot_date.asc,sku.asc"},
            timeout=120,
        )
        r.raise_for_status()
        batch = r.json()
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
