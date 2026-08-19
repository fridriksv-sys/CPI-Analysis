"""Test the fuel-augmented h=1 nowcast: current month + backtest vs model-only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
from vnv import airfares, ingest, models, nowcast

spl = ingest.load_sub_spliced()
new = ingest.load_sub_new()
old = ingest.load_panel_old()
head = ingest.load_headline()

g = models.build_component_history(spl, old, new)
last_m = g.index.max()
latest = new[new.manudur == last_m].set_index("code")
w0 = latest.loc[models.COMPONENTS, "vaegi"]
print("components:", len(models.COMPONENTS), " weight sum:", round(w0.sum(), 2))

cal = nowcast.calibrate_fuel()
print(f"fuel calibration: alpha={cal.alpha:+.3f} beta={cal.beta:.3f} "
      f"resid_sd={cal.resid_sd:.3f}pp n={cal.n_obs}")
acal = nowcast.calibrate_airfares()
print("airfare calibration:", "PENDING (need >=2 collection windows)" if acal is None
      else f"beta={acal.beta:.3f} n={acal.n_obs}")
print("airfare scraped index:", "empty" if airfares.airfare_index_mm().empty
      else airfares.airfare_index_mm().to_dict())

# current-month nowcast
fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
fc = pd.DataFrame({c: models.forecast_component(fits[c], last_m, 1) for c in models.COMPONENTS})
obs = nowcast.fuel_nowcast(last_m + 1, cal)
fc_obs = nowcast.apply_observables(fc, obs)
hm_model, _, _ = models.aggregate_bottom_up(fc, w0)
hm_obs, _, _ = models.aggregate_bottom_up(fc_obs, w0)
print(f"\nnowcast {last_m + 1}: model-only={hm_model.iloc[0]:+.3f}  "
      f"with fuel observed={hm_obs.iloc[0]:+.3f}  (fuel m/m obs: {obs['CP0722']:+.2f})")

# backtest: h=1 with vs without the fuel observable, 2025-02..last
rows = []
for m in [m for m in g.index if m > pd.Period("2025-01", "M")]:
    g_tr = g[g.index < m]
    w_prev = new[(new.manudur == m - 1) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
    if len(w_prev) < len(models.COMPONENTS):
        continue
    f = {c: models.forecast_component(models.fit_component(g_tr[c], c), m - 1, 1).iloc[0]
         for c in models.COMPONENTS}
    fmm = pd.DataFrame([f], index=[m])
    hm, _, _ = models.aggregate_bottom_up(fmm, w_prev)
    try:
        cal_m = nowcast.calibrate_fuel(train_end=m - 1)
        fmm_o = nowcast.apply_observables(fmm, nowcast.fuel_nowcast(m, cal_m))
    except ValueError:
        fmm_o = fmm
    hmo, _, _ = models.aggregate_bottom_up(fmm_o, w_prev)
    rows.append({"manudur": m, "model": hm.iloc[0], "model+fuel": hmo.iloc[0],
                 "actual": head[("CPI", "change_M")].get(m, np.nan)})
bt = pd.DataFrame(rows).set_index("manudur").dropna()
print(f"\nbacktest n={len(bt)}")
for col in ["model", "model+fuel"]:
    rmse = np.sqrt(((bt[col] - bt.actual) ** 2).mean())
    print(f"{col:12s} RMSE={rmse:.4f}  MAE={(bt[col] - bt.actual).abs().mean():.4f}")
print(bt.round(3).to_string())
