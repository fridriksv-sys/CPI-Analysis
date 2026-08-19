"""Explore the HMS rent index -> Hagstofa reiknuð húsaleiga (CP042) mapping."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import hms, ingest

rent = hms.load_rent()
house = hms.load_house()
spl = ingest.load_sub_spliced()
sub_new = ingest.load_sub_new()

# Target: reiknuð húsaleiga CP042 (spliced history + new panel)
cp042 = spl[spl.code == "CP042"].set_index("manudur").visitala
cp042_new = sub_new[sub_new.code == "CP042"].set_index("manudur")
print("CP042 (spliced) coverage:", cp042.index.min(), "-", cp042.index.max())
print("HMS rent coverage:", rent.index.min(), "-", rent.index.max())

# m/m changes
tgt = (cp042.pct_change() * 100).rename("cp042_mm")
drv = (rent.leiguvisitala.pct_change() * 100).rename("hms_rent_mm")
BREAK = pd.Period("2024-07", "M")

df = pd.concat([tgt, drv], axis=1).dropna()
post = df[df.index >= BREAK]
print(f"\noverlap months (rent available): {len(df)}  post-break: {len(post)}")

print("\n=== contemporaneous + lagged correlations (post-June-2024) ===")
for lag in range(0, 5):
    d = pd.concat([post.cp042_mm, post.hms_rent_mm.shift(lag)], axis=1).dropna()
    if len(d) > 3:
        print(f"  lag {lag}: corr={d.cp042_mm.corr(d.hms_rent_mm):+.3f}  n={len(d)}")

# smoothing: stock turns over slowly -> CP042 should track a MA of HMS rent
print("\n=== CP042 mm vs trailing-mean HMS rent mm (post-break) ===")
for win in [1, 2, 3, 6, 12]:
    ma = post.hms_rent_mm.rolling(win).mean()
    d = pd.concat([post.cp042_mm, ma], axis=1).dropna()
    if len(d) > 3:
        b = np.polyfit(d.iloc[:, 1], d.cp042_mm, 1)
        resid = d.cp042_mm - np.polyval(b, d.iloc[:, 1])
        r2 = 1 - (resid**2).sum() / ((d.cp042_mm - d.cp042_mm.mean())**2).sum()
        print(f"  MA{win}: slope={b[0]:+.3f} const={b[1]:+.3f} R2={r2:.3f} "
              f"residSD={resid.std():.3f} n={len(d)}")

# level ratio: is CP042 ~ a smoothed level of HMS rent?
print("\n=== recent levels (both rebased to 2024-07=100) ===")
base = pd.Period("2024-07", "M")
lv = pd.DataFrame({
    "CP042": cp042 / cp042[base] * 100,
    "HMS_rent": rent.leiguvisitala / rent.leiguvisitala[base] * 100,
}).dropna()
print(lv.tail(14).round(2).to_string())

# persistence of CP042 itself (post-break)
print("\n=== CP042 m/m persistence (post-break) ===")
p = post.cp042_mm
print("mean:", round(p.mean(), 3), "sd:", round(p.std(), 3),
      "AR(1):", round(p.autocorr(1), 3) if len(p) > 3 else "n/a")
print(post.round(3).to_string())
