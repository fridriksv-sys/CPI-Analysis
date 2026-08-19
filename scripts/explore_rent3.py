"""Post-break-only rent models: adaptive mean (EWMA) +/- HMS tilt, OOS."""
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
y = (cp042.pct_change() * 100).rename("y")
x = (rent.leiguvisitala.pct_change() * 100).rename("x")
BREAK = pd.Period("2024-07", "M")

d = pd.concat([y, x], axis=1)
d["x_lag13"] = (d.x.shift(1) + d.x.shift(2) + d.x.shift(3)) / 3
d = d[d.index >= BREAK].dropna(subset=["y"])

def ewma(vals, halflife):
    a = 1 - 0.5 ** (1 / halflife)
    m = vals[0]
    for v in vals[1:]:
        m = a * v + (1 - a) * m
    return m

# OOS: train only on post-break data up to t-1, predict t. min_train=6.
rows = []
idx = d.index
for i in range(6, len(idx)):
    m = idx[i]
    tr = d.iloc[:i]
    preds = {
        "rw": tr.y.iloc[-1],
        "drift": tr.y.mean(),
        "ewma6": ewma(tr.y.values, 6),
        "ewma3": ewma(tr.y.values, 3),
    }
    # AR(1)
    yy = tr.y.values
    preds["ar1"] = (np.polyval(np.polyfit(yy[:-1], yy[1:], 1), tr.y.iloc[-1])
                    if len(yy) > 3 and np.var(yy[:-1]) > 0 else tr.y.mean())
    # EWMA + HMS tilt: shrink a regression on x_lag13 toward the ewma
    xl = d.x_lag13.iloc[i]
    trx = tr.dropna(subset=["x_lag13"])
    if len(trx) > 6 and np.var(trx.x_lag13) > 0 and not np.isnan(xl):
        bb = np.polyfit(trx.x_lag13, trx.y, 1)
        reg = np.polyval(bb, xl)
        preds["ewma6+hms"] = 0.7 * preds["ewma6"] + 0.3 * reg
    else:
        preds["ewma6+hms"] = preds["ewma6"]
    rows.append({"manudur": m, "actual": d.y.iloc[i], **preds})

bt = pd.DataFrame(rows).set_index("manudur").dropna()
print(f"post-break OOS backtest n={len(bt)} ({bt.index.min()}..{bt.index.max()})")
print(f"(series m/m SD = {d.y.std():.3f}; a perfect-mean predictor ~ that RMSE)")
res = {}
for col in ["rw", "ar1", "drift", "ewma6", "ewma3", "ewma6+hms"]:
    rmse = np.sqrt(((bt[col] - bt.actual)**2).mean())
    mae = (bt[col] - bt.actual).abs().mean()
    res[col] = rmse
    print(f"  {col:10s} RMSE={rmse:.4f}  MAE={mae:.4f}")
best = min(res, key=res.get)
print(f"\nbest: {best} (RMSE {res[best]:.4f}) vs RW {res['rw']:.4f}, AR1 {res['ar1']:.4f}")
print(bt.round(3).to_string())
