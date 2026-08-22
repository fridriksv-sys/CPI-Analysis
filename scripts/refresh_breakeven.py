"""Refresh the market breakeven (RIKB−RIKS) from lanamal.is into the slot.

    .venv\\Scripts\\python.exe scripts\\refresh_breakeven.py

Writes data/benchmarks/breakeven.csv (the slot report / model_vs_breakeven read).
Run on whatever cadence you compare against the market — daily during active
analysis, or ad hoc before generating a monthly report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnv import lanamal

print(lanamal.breakeven_curve(use_cache=False).to_string(index=False))
print("\nwrote:", lanamal.update_breakeven_slot(use_cache=False))
