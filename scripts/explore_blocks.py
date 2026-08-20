"""Estimate FX/food/wage pass-through for candidate block components."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import ingest, sedlabanki, worldfood

spl = ingest.load_sub_spliced()
fx = sedlabanki.fx_mm()          # observable at lag 0 (mid-month sample)
food = worldfood.food_mm()       # observable at lag 0
wage = ingest.wages_mm()         # observable at lag 1+

def comp_mm(code):
    return (spl[spl.code == code].set_index("manudur").visitala.pct_change() * 100).rename(code)

def ardl_passthrough(y, driver, lags=(0, 1, 2, 3), sample_start="2020-01"):
    """Cumulative pass-through = sum of distributed-lag coefficients (with seasonal)."""
    d = pd.concat([y.rename("y")] + [driver.shift(k).rename(f"d{k}") for k in lags], axis=1)
    d = d[d.index >= sample_start].dropna()
    if len(d) < 24:
        return None
    # de-seasonalize y by monthly means
    d["mon"] = d.index.month
    seas = d.groupby("mon").y.transform("mean")
    yy = (d.y - seas).values
    X = np.column_stack([np.ones(len(d))] + [d[f"d{k}"].values for k in lags])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    r2 = 1 - (resid**2).sum() / ((yy - yy.mean())**2).sum()
    return {"passthrough": beta[1:].sum(), "coefs": beta[1:].round(3).tolist(),
            "r2": r2, "resid_sd": resid.std(), "n": len(d)}

print("=== FX pass-through (imported goods + food), 2020- ===")
for code in ["CP03", "CP05", "CP071", "CP08", "CP01", "CP02"]:
    r = ardl_passthrough(comp_mm(code), fx)
    if r:
        print(f"  {code}: 12m passthrough={r['passthrough']:+.3f}  coefs(l0-3)={r['coefs']}  "
              f"R2={r['r2']:.2f}  residSD={r['resid_sd']:.2f}  n={r['n']}")

print("\n=== food: FX + FAO world food ===")
for code in ["CP01"]:
    y = comp_mm(code)
    d = pd.concat([y.rename("y"), fx.rename("fx"), fx.shift(1).rename("fx1"),
                   food.rename("fao"), food.shift(1).rename("fao1")], axis=1)
    d = d[d.index >= "2020-01"].dropna()
    d["mon"] = d.index.month
    yy = (d.y - d.groupby("mon").y.transform("mean")).values
    X = np.column_stack([np.ones(len(d)), d.fx, d.fx1, d.fao, d.fao1])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    r2 = 1 - (resid**2).sum() / ((yy - yy.mean())**2).sum()
    print(f"  {code}: fx(l0,l1)={beta[1]:+.3f},{beta[2]:+.3f}  fao(l0,l1)={beta[3]:+.3f},{beta[4]:+.3f}  "
          f"R2={r2:.2f} residSD={resid.std():.2f} n={len(d)}")

print("\n=== domestic services: wage m/m (lag 1-3) ===")
for code in ["CP11", "CP043", "CP06", "CP10", "CP13"]:
    y = comp_mm(code)
    wl = (wage.shift(1) + wage.shift(2) + wage.shift(3)) / 3
    d = pd.concat([y.rename("y"), wl.rename("w")], axis=1)
    d = d[d.index >= "2016-01"].dropna()
    d["mon"] = d.index.month
    yy = (d.y - d.groupby("mon").y.transform("mean")).values
    X = np.column_stack([np.ones(len(d)), d.w.values])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    r2 = 1 - (resid**2).sum() / ((yy - yy.mean())**2).sum()
    print(f"  {code}: wage slope={beta[1]:+.3f}  R2={r2:.2f}  residSD={resid.std():.2f}  n={len(d)}")
