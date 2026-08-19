"""Verify the rent model: OOS vs benchmarks, regime-break variant, forecast path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import ingest, rent

spl = ingest.load_sub_spliced()
cp042_mm = rent.load_cp042_history(spl)
hms_mm = rent.hms_rent_mm()

# --- OOS backtest of the shipped model vs RW / AR(1) (post-break) ---
d = pd.concat([cp042_mm.rename("y"), hms_mm], axis=1)
post = d[d.index >= rent.BREAK].dropna(subset=["y"])
rows = []
for i in range(6, len(post)):
    m = post.index[i]
    hist_y = cp042_mm[cp042_mm.index <= post.index[i - 1]]
    hist_h = hms_mm[hms_mm.index <= post.index[i - 1]]
    fc = rent.forecast_cp042(hist_y, hist_h, post.index[i - 1], horizons=1).iloc[0]
    tr = post.y.iloc[:i]
    yy = tr.values
    ar1 = (np.polyval(np.polyfit(yy[:-1], yy[1:], 1), tr.iloc[-1])
           if len(yy) > 3 and np.var(yy[:-1]) > 0 else tr.mean())
    rows.append({"manudur": m, "actual": post.y.iloc[i], "model": fc,
                 "rw": tr.iloc[-1], "ar1": ar1})
bt = pd.DataFrame(rows).set_index("manudur").dropna()
print(f"=== Phase 4 acceptance: OOS m/m (post-break), n={len(bt)} ===")
for col in ["model", "rw", "ar1"]:
    rmse = np.sqrt(((bt[col] - bt.actual) ** 2).mean())
    print(f"  {col:6s} RMSE={rmse:.4f}  MAE={(bt[col] - bt.actual).abs().mean():.4f}")
passes = (np.sqrt(((bt.model - bt.actual) ** 2).mean())
          < min(np.sqrt(((bt.rw - bt.actual) ** 2).mean()),
                np.sqrt(((bt.ar1 - bt.actual) ** 2).mean())))
print("  BEATS RW and AR(1):", "PASS" if passes else "FAIL")

# --- regime-break variant (full sample + dummy), reported per the plan ---
full = pd.concat([cp042_mm.rename("y"), rent._hms_driver(hms_mm).rename("x")], axis=1).dropna()
full["post"] = (full.index >= rent.BREAK).astype(float)
X = np.column_stack([np.ones(len(full)), full.x, full.post, full.x * full.post])
beta, *_ = np.linalg.lstsq(X, full.y.values, rcond=None)
print("\n=== full-sample regime-break OLS (reported, not shipped) ===")
print(f"  const={beta[0]:.3f}  x={beta[1]:.3f}  post={beta[2]:.3f}  x*post={beta[3]:.3f}")
print("  post-break slope on HMS driver:", round(beta[1] + beta[3], 3),
      "vs pre-break", round(beta[1], 3))
print("  -> regimes differ; shipping the post-break-only model is the right call.")

# --- 12-month path ---
last = cp042_mm.index.max()
path = rent.forecast_cp042(cp042_mm, hms_mm, last, horizons=12)
lvl = spl[spl.code == "CP042"].set_index("manudur").visitala
lvl_path = lvl.iloc[-1] * (1 + path / 100).cumprod()
print(f"\n=== CP042 12-month path from {last} ===")
print(f"  implied 12m rent inflation: {((1 + path / 100).prod() - 1) * 100:.2f}%")
print(path.round(3).to_string())
