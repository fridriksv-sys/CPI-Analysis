"""Full h=1 headline backtest: generic -> +fuel -> +rent -> +FX blocks (Phase 5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
from vnv import blocks, ingest, models, nowcast, rent, sedlabanki

spl = ingest.load_sub_spliced()
new = ingest.load_sub_new()
old = ingest.load_panel_old()
head = ingest.load_headline()
g = models.build_component_history(spl, old, new)
fx = sedlabanki.fx_mm()
hms_mm = rent.hms_rent_mm()
FX_COMPS = [c for cs in blocks.fx_components().values() for c in cs]

rows = []
for m in [t for t in g.index if t > pd.Period("2025-01", "M")]:
    g_tr = g[g.index < m]
    w_prev = new[(new.manudur == m - 1) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
    if len(w_prev) < len(models.COMPONENTS):
        continue
    fits = {c: models.fit_component(g_tr[c], c) for c in models.COMPONENTS}

    # 1) generic
    fc0 = pd.DataFrame({c: models.forecast_component(fits[c], m - 1, 1) for c in models.COMPONENTS})
    h0, _, _ = models.aggregate_bottom_up(fc0, w_prev)
    # 2) +fuel observed
    cal = nowcast.calibrate_fuel(train_end=m - 1)
    fc1 = nowcast.apply_observables(fc0.copy(), nowcast.fuel_nowcast(m, cal))
    h1, _, _ = models.aggregate_bottom_up(fc1, w_prev)
    # 3) +rent model
    fc2 = models.forecast_components(fits, m - 1, 1, hms_rent_mm=hms_mm[hms_mm.index < m],
                                     sub_spliced=spl)
    fc2 = nowcast.apply_observables(fc2, nowcast.fuel_nowcast(m, cal))
    h2, _, _ = models.aggregate_bottom_up(fc2, w_prev)
    # 4) +FX blocks (full stack)
    fc3 = models.forecast_components(fits, m - 1, 1, hms_rent_mm=hms_mm[hms_mm.index < m],
                                     sub_spliced=spl, comp_history=g_tr, fx_mm=fx)
    fc3 = nowcast.apply_observables(fc3, nowcast.fuel_nowcast(m, cal))
    h3, _, _ = models.aggregate_bottom_up(fc3, w_prev)

    rows.append({"m": m, "actual": head[("CPI", "change_M")].get(m, np.nan),
                 "generic": h0.iloc[0], "+fuel": h1.iloc[0], "+rent": h2.iloc[0], "+fx": h3.iloc[0]})

bt = pd.DataFrame(rows).set_index("m").dropna()
ex_jan = bt.drop(pd.Period("2026-01", "M"), errors="ignore")
print(f"Full h=1 headline backtest, n={len(bt)}\n")
print(f"{'model':10s} {'RMSE all':>9s} {'MAE all':>8s} {'RMSE exJan':>11s} {'MAE exJan':>10s}")
for col in ["generic", "+fuel", "+rent", "+fx"]:
    r = np.sqrt(((bt[col] - bt.actual) ** 2).mean())
    ma = (bt[col] - bt.actual).abs().mean()
    rj = np.sqrt(((ex_jan[col] - ex_jan.actual) ** 2).mean())
    mj = (ex_jan[col] - ex_jan.actual).abs().mean()
    print(f"{col:10s} {r:9.4f} {ma:8.4f} {rj:11.4f} {mj:10.4f}")
print("\nJan-2026 is the fiscal-shock outlier (km-charge +1.01pp); it needs the")
print("dated fiscal_calendar applied, not more model. Ex-Jan shows the structural gain.")
print("h=1 <=0.15pp target remains airfare-gated (airfares still collecting).")
print("\n" + bt.round(3).to_string())
