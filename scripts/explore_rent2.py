"""Distributed-lag rent model with observable (lagged) HMS rent + OOS backtest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import hms, ingest

rent = hms.load_rent()
spl = ingest.load_sub_spliced()
cp042 = spl[spl.code == "CP042"].set_index("manudur").visitala
tgt = (cp042.pct_change() * 100).rename("y")
drv = (rent.leiguvisitala.pct_change() * 100).rename("x")
BREAK = pd.Period("2024-07", "M")

# Observable driver: mean of HMS rent m/m over lags 1..3 (all known by forecast time)
d = pd.concat([tgt, drv], axis=1)
d["x_lag13"] = (d.x.shift(1) + d.x.shift(2) + d.x.shift(3)) / 3
d = d.dropna(subset=["y", "x_lag13"])
post = d[d.index >= BREAK]
print("post-break obs:", len(post))

# Full-sample fit for reference
b = np.polyfit(post.x_lag13, post.y, 1)
resid = post.y - np.polyval(b, post.x_lag13)
r2 = 1 - (resid**2).sum() / ((post.y - post.y.mean())**2).sum()
print(f"in-sample: y = {b[1]:.3f} + {b[0]:.3f}*mean(HMS_mm, lag1-3)  R2={r2:.3f}  residSD={resid.std():.3f}")

# --- expanding-window OOS backtest, min 12 training obs ---
def backtest(series_y, series_x, min_train=12):
    rows = []
    idx = series_y.index
    for i in range(min_train, len(idx)):
        m = idx[i]
        tr_y = series_y.iloc[:i]
        tr_x = series_x.iloc[:i]
        # model: distributed lag
        bb = np.polyfit(tr_x, tr_y, 1)
        pred_model = np.polyval(bb, series_x.iloc[i])
        # RW on m/m: last month's y
        pred_rw = tr_y.iloc[-1]
        # AR(1) on y
        yy = tr_y.values
        if len(yy) > 3 and np.var(yy[:-1]) > 0:
            phi = np.polyfit(yy[:-1], yy[1:], 1)
            pred_ar = np.polyval(phi, tr_y.iloc[-1])
        else:
            pred_ar = tr_y.mean()
        # drift = expanding mean
        pred_drift = tr_y.mean()
        rows.append({"manudur": m, "actual": series_y.iloc[i], "model": pred_model,
                     "rw": pred_rw, "ar1": pred_ar, "drift": pred_drift})
    return pd.DataFrame(rows).set_index("manudur")

# Use ALL overlap (from 2023-08 once lags available) so training has >=12 before post-break
full = d.dropna(subset=["y", "x_lag13"])
bt = backtest(full.y, full.x_lag13, min_train=12)
bt = bt.dropna()
print(f"\nOOS backtest n={len(bt)} ({bt.index.min()}..{bt.index.max()})")
for col in ["model", "rw", "ar1", "drift"]:
    rmse = np.sqrt(((bt[col] - bt.actual)**2).mean())
    mae = (bt[col] - bt.actual).abs().mean()
    print(f"  {col:6s} RMSE={rmse:.4f}  MAE={mae:.4f}")
print(bt.round(3).to_string())
