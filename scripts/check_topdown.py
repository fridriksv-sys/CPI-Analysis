"""Phase 6: top-down h=12 y/y backtest + MinT reconciliation properties."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
from vnv import ingest, models, reconcile, rent, sedlabanki, topdown

head = ingest.load_headline()
mm = head[("CPI", "change_M")].dropna()
idx = head[("CPI", "index")].dropna()
yy = head[("CPI", "change_A")].dropna()

# --- h=12 y/y backtest: top-down vs RW-on-y/y ---
def yoy_from_path(path_mm):
    return ((1 + path_mm / 100).prod() - 1) * 100

rows = []
for jo in [t for t in mm.index if pd.Period("2016-06", "M") <= t <= mm.index[-1] - 12]:
    f = topdown.fit(mm[mm.index <= jo])
    path = topdown.forecast(f, 12)
    target_month = jo + 12
    if target_month not in yy.index:
        continue
    rows.append({"jump_off": jo, "actual_yoy": yy[target_month],
                 "topdown_yoy": yoy_from_path(path), "rw_yoy": yy[jo]})
bt = pd.DataFrame(rows).set_index("jump_off").dropna()
print(f"=== h=12 y/y backtest, n={len(bt)} ({bt.index.min()}..{bt.index.max()}) ===")
for col in ["topdown_yoy", "rw_yoy"]:
    rmse = np.sqrt(((bt[col] - bt.actual_yoy) ** 2).mean())
    mae = (bt[col] - bt.actual_yoy).abs().mean()
    print(f"  {col:12s} RMSE={rmse:.3f}  MAE={mae:.3f}")
beats = (np.sqrt(((bt.topdown_yoy - bt.actual_yoy) ** 2).mean())
         < np.sqrt(((bt.rw_yoy - bt.actual_yoy) ** 2).mean()))
print("  TOP-DOWN BEATS RW-on-y/y at h=12:", "PASS" if beats else "FAIL")

# exclude the 2021-23 inflation surge (RW-on-yoy is unbeatable mid-ramp) to show
# the structural read:
calm = bt[(bt.index < "2021-01") | (bt.index >= "2023-06")]
print(f"\n  ex-surge (n={len(calm)}): topdown RMSE="
      f"{np.sqrt(((calm.topdown_yoy-calm.actual_yoy)**2).mean()):.3f}  "
      f"rw RMSE={np.sqrt(((calm.rw_yoy-calm.actual_yoy)**2).mean()):.3f}")

# --- reconciliation on the current path: identity + pull toward top-down ---
spl = ingest.load_sub_spliced(); new = ingest.load_sub_new(); old = ingest.load_panel_old()
g = models.build_component_history(spl, old, new)
last_m = g.index.max()
w0 = new[(new.manudur == last_m) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
fc = models.forecast_components(fits, last_m, 12, hms_rent_mm=rent.hms_rent_mm(),
                                sub_spliced=spl, comp_history=g, fx_mm=sedlabanki.fx_mm())
contrib, wpath = reconcile.contributions_from_forecast(fc, w0)
bu_head = contrib.sum(axis=1)

tf = topdown.fit(mm)
td_path = topdown.forecast(tf, 12)
td_sd = topdown.error_sd_by_horizon(tf, 12)
# per-component h=1 contribution error SD ~ resid_sd(component)*weight/100
csd1 = pd.Series({c: fits[c].resid.std() * float(w0.get(c, 0)) / 100 for c in fc.columns})
rec_head, rec_contrib, rec_sd = reconcile.reconcile_path(contrib, td_path, csd1, td_sd)

cmp = pd.DataFrame({"bottom_up": bu_head, "top_down": td_path, "reconciled": rec_head})
print("\n=== path reconciliation (m/m %) ===")
print(cmp.round(3).to_string())
ident = (rec_contrib.sum(axis=1) - rec_head).abs().max()
print(f"\naggregation identity |Σcontrib - headline| max = {ident:.2e} (exact by construction)")
print(f"bottom-up 12m infl:  {yoy_from_path(bu_head):.2f}%")
print(f"top-down 12m infl:   {yoy_from_path(td_path):.2f}%")
print(f"reconciled 12m infl: {yoy_from_path(rec_head):.2f}%")
pull_h1 = abs(rec_head.iloc[0] - bu_head.iloc[0])
pull_h12 = abs(rec_head.iloc[-1] - bu_head.iloc[-1])
print(f"reconciliation move from bottom-up: h=1 {pull_h1:.3f}pp -> h=12 {pull_h12:.3f}pp "
      f"({'grows' if pull_h12 > pull_h1 else 'shrinks'} with horizon, as designed)")
