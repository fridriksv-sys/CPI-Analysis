"""Diagnose chain-link months and the weights table missing-value code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

pd.set_option("display.width", 220)
from vnv import ingest, reconstruct

wnew = ingest.load_weights_new()
print("=== VIS01306 division rows ===")
print(wnew[wnew.code.isin(reconstruct.DIV_NEW + ["CP00"])].to_string())

old = ingest.load_panel_old()
div = old[old.code.isin(reconstruct.DIV_OLD)].copy().sort_values(["code", "manudur"])
div["vaegi_lag"] = div.groupby("code")["vaegi"].shift(1)
pub = old[old.code == "IS00"].set_index("manudur")["manadarbreyting"]

for m in ["2019-04", "2020-04", "2021-04", "2022-04", "2023-04", "2024-04", "2025-01", "2019-03", "2025-02"]:
    p = pd.Period(m, "M")
    d = div[div.manudur == p]
    if d.empty:
        continue
    rec_lag = (d.vaegi_lag * d.manadarbreyting).sum() / 100
    rec_t = (d.vaegi * d.manadarbreyting).sum() / 100
    ah = d_ah = old[(old.manudur == p) & old.code.isin(reconstruct.DIV_OLD)].ahrif.sum()
    print(f"{m}: published={pub.get(p)}  vaegi(t-1)={rec_lag:.4f}  vaegi(t)={rec_t:.4f}  sum_ahrif={ah:.2f}")

# check what vaegi does across March/April rebase: does it jump discontinuously in April?
w = div.pivot_table(index="manudur", columns="code", values="vaegi")
print("\nIS01 vaegi around rebases:")
print(w["IS01"]["2019-01":"2019-06"].to_string())
print(w["IS01"]["2024-11":"2025-03"].to_string())
