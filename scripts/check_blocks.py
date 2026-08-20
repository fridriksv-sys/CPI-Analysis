"""Does the FX tilt improve component + headline forecasts OOS? (Phase 5)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
from vnv import blocks, ingest, models, sedlabanki

spl = ingest.load_sub_spliced()
new = ingest.load_sub_new()
old = ingest.load_panel_old()
head = ingest.load_headline()
fx = sedlabanki.fx_mm()
g = models.build_component_history(spl, old, new)

FX_COMPS = [c for cs in blocks.fx_components().values() for c in cs]
print("FX-driven components:", FX_COMPS)

# --- per-component OOS: generic seasonal+AR vs +FX tilt ---
print("\n=== per-component OOS RMSE (expanding, 2023-01+) ===")
for code in FX_COMPS:
    y = g[code].dropna()
    rows = []
    for m in [t for t in y.index if t >= pd.Period("2023-01", "M")]:
        tr = y[y.index < m]
        fit_g = models.fit_component(tr, code)
        pred_g = models.forecast_component(fit_g, m - 1, 1).iloc[0]
        fxfit = blocks.fit_fx_passthrough(tr, fx, train_end=m - 1)
        tilt = blocks.fx_tilt_forecast(fxfit, fx, m)
        if tilt is not None and fxfit is not None:
            seas = fxfit["seasonal"].get(m.month, tr[tr.index.month == m.month].mean())
            pred_fx = (1 - blocks.FX_TILT) * pred_g + blocks.FX_TILT * (seas + tilt)
        else:
            pred_fx = pred_g
        rows.append({"m": m, "actual": y.get(m, np.nan), "generic": pred_g, "fx": pred_fx})
    d = pd.DataFrame(rows).dropna()
    rg = np.sqrt(((d.generic - d.actual) ** 2).mean())
    rf = np.sqrt(((d.fx - d.actual) ** 2).mean())
    flag = "better" if rf < rg else "worse"
    print(f"  {code}: generic RMSE={rg:.3f}  +FX={rf:.3f}  ({flag}, passthrough "
          f"{blocks.fit_fx_passthrough(y, fx)['passthrough']:+.2f})")

# --- headline h=1 backtest: model+fuel vs +FX blocks ---
print("\n=== headline h=1 backtest ===")
rows = []
for m in [t for t in g.index if t > pd.Period("2025-01", "M")]:
    g_tr = g[g.index < m]
    w_prev = new[(new.manudur == m - 1) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
    if len(w_prev) < len(models.COMPONENTS):
        continue
    base, fxb = {}, {}
    for c in models.COMPONENTS:
        fit_g = models.fit_component(g_tr[c], c)
        pg = models.forecast_component(fit_g, m - 1, 1).iloc[0]
        base[c] = pg
        if c in FX_COMPS:
            fxfit = blocks.fit_fx_passthrough(g_tr[c], fx, train_end=m - 1)
            tilt = blocks.fx_tilt_forecast(fxfit, fx, m)
            if tilt is not None and fxfit is not None:
                seas = fxfit["seasonal"].get(m.month, g_tr[c][g_tr[c].index.month == m.month].mean())
                fxb[c] = (1 - blocks.FX_TILT) * pg + blocks.FX_TILT * (seas + tilt)
            else:
                fxb[c] = pg
        else:
            fxb[c] = pg
    hb, _, _ = models.aggregate_bottom_up(pd.DataFrame([base], index=[m]), w_prev)
    hf, _, _ = models.aggregate_bottom_up(pd.DataFrame([fxb], index=[m]), w_prev)
    rows.append({"m": m, "actual": head[("CPI", "change_M")].get(m, np.nan),
                 "generic": hb.iloc[0], "with_fx": hf.iloc[0]})
bt = pd.DataFrame(rows).set_index("m").dropna()
for col in ["generic", "with_fx"]:
    print(f"  {col:8s} RMSE={np.sqrt(((bt[col]-bt.actual)**2).mean()):.4f}  "
          f"MAE={(bt[col]-bt.actual).abs().mean():.4f}")
print(f"  (n={len(bt)})")
