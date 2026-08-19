"""Smoke-test the v0 forecast pipeline + walk-forward h=1 backtest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import ingest, models

spl = ingest.load_sub_spliced()
new = ingest.load_sub_new()
old = ingest.load_panel_old()
head = ingest.load_headline()

g = models.build_component_history(spl, old, new)
print("history coverage:", g.index.min(), "-", g.index.max())
print("NaN count per component (2016+):")
print(g["2016-01":].isna().sum().to_string())

# jump-off weights: latest published vaegi for these codes
last_m = new.manudur.max()
w0 = new[(new.manudur == last_m) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
print("\njump-off month:", last_m, " weight sum:", w0.sum().round(2))
print(w0.round(2).to_string())

# fit + forecast
fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
fc = pd.DataFrame({c: models.forecast_component(fits[c], last_m, 12) for c in models.COMPONENTS})
head_mm, contribs, wpath = models.aggregate_bottom_up(fc, w0)
print("\n12-month headline m/m forecast (%):")
print(head_mm.round(3).to_string())
print("\nimplied 12m inflation (%):", round(((1 + head_mm / 100).prod() - 1) * 100, 2))

# --- walk-forward h=1 backtest over last 24 months ---
test_months = g.index[(g.index > pd.Period("2024-07", "M"))]
test_months = [m for m in test_months if m <= last_m]
rows = []
for m in test_months:
    g_train = g[g.index < m]
    w_prev = new[(new.manudur == m - 1) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
    if len(w_prev) < len(models.COMPONENTS):
        continue  # no published weights before 2025
    f = {}
    for c in models.COMPONENTS:
        fit = models.fit_component(g_train[c], c)
        f[c] = models.forecast_component(fit, m - 1, 1).iloc[0]
    fmm = pd.DataFrame([f], index=[m])
    hm, _, _ = models.aggregate_bottom_up(fmm, w_prev)
    # actual = the published headline m/m (VIS01000), the series that settles bonds
    seas = {c: g_train[c][g_train.index.month == m.month]["2016":].mean() for c in models.COMPONENTS}
    hs, _, _ = models.aggregate_bottom_up(pd.DataFrame([seas], index=[m]), w_prev)
    rows.append({
        "manudur": m, "model": hm.iloc[0], "seasonal_naive": hs.iloc[0],
        "rw": g.loc[m - 1] @ (w_prev / w_prev.sum()) if (m - 1) in g.index else np.nan,
        "actual": head[("CPI", "change_M")].get(m, np.nan),
    })
bt = pd.DataFrame(rows).set_index("manudur").dropna()
print("\n=== walk-forward h=1 backtest ===")
print(bt.round(3).to_string())
for col in ["model", "seasonal_naive", "rw"]:
    rmse = np.sqrt(((bt[col] - bt.actual) ** 2).mean())
    mae = (bt[col] - bt.actual).abs().mean()
    print(f"{col:15s} RMSE={rmse:.4f}  MAE={mae:.4f}")
