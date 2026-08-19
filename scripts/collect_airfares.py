"""Daily airfare collection for the CP073 nowcast panel.

Safe to schedule every day: it self-limits to Hagstofa's collection window
(days 1-15) unless --force is passed, so a plain daily trigger only collects
when the data has CPI value.

    .venv\\Scripts\\python.exe scripts\\collect_airfares.py [--force]
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnv import airfares

if date.today().day > 15 and "--force" not in sys.argv:
    print(f"day {date.today().day} is outside the collection window (1-15); skipping.")
    sys.exit(0)

df = airfares.collect()
if df.empty:
    print("no fares collected")
    sys.exit(1)
print(f"collected {len(df)} lowest fares across {df.dest.nunique()} destinations")
print(df[["dest", "flight_type", "depart_date", "return_date", "fare_isk"]].to_string(index=False))
