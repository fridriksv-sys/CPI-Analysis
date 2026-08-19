"""Calibrate Gasvaktin collection-window prices against published fuel subindices."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import fuel, ingest

bensin = fuel.scraped_mm("bensin95")
diesel = fuel.scraped_mm("diesel")
print("scraped m/m coverage:", bensin.index.min(), "-", bensin.index.max())

new = ingest.load_sub_new()
old = ingest.load_panel_old()

pub_b = new[new.code == "CP07222"].set_index("manudur").manadarbreyting.rename("pub")
pub_d = new[new.code == "CP07221"].set_index("manudur").manadarbreyting.rename("pub")
# long calibration: old IS0722 eldsneyti (bensin+diesel), 2016-2025
pub_e = old[old.code == "IS0722"].set_index("manudur").manadarbreyting.rename("pub")
w_bd = 2.0 / 3.0  # rough bensin share of the fuel subindex (1.7 vs 0.8 weight)
mix = (w_bd * bensin + (1 - w_bd) * diesel).rename("scraped")


def calib(scraped: pd.Series, pub: pd.Series, label: str):
    df = pd.concat([scraped.rename("scraped"), pub], axis=1).dropna()
    x, y = df.scraped, df.pub
    beta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    alpha = y.mean() - beta * x.mean()
    resid = y - (alpha + beta * x)
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"{label:28s} n={len(df):3d}  beta={beta:5.3f}  alpha={alpha:+5.3f}  "
          f"R2={r2:.3f}  resid SD={resid.std():.3f}pp")
    return df

print("\n=== calibration: scraped collection-window m/m vs published subindex m/m ===")
d1 = calib(mix, pub_e[pub_e.index >= "2016-06"], "IS0722 eldsneyti 2016-2025")
d2 = calib(bensin, pub_b, "CP07222 bensin 2025-")
d3 = calib(diesel, pub_d, "CP07221 diesel 2025-")

print("\nlast 8 months, bensin (scraped vs published):")
print(pd.concat([bensin.rename("scraped"), pub_b], axis=1).dropna().tail(8).round(2).to_string())
print("\ncurrent-month nowcast inputs (partial window):")
print("bensin m/m:", round(bensin.iloc[-1], 2), " diesel m/m:", round(diesel.iloc[-1], 2),
      "for", bensin.index[-1])
