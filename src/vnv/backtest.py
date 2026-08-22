"""Pseudo-real-time backtest and benchmarks (Phase 7).

Expanding-window, refit each jump-off on data through t-1, aggregated with the
weights published at t-1 (no look-ahead). Reports RMSE/MAE by horizon and a
contribution-level h=1 error decomposition so it is clear WHICH block drives
misses. Benchmarks: seasonal-naive on m/m and random-walk-on-y/y (the plan's
floors); analyst / Peningamál / breakeven are loaded from data slots (benchmarks.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import blocks, ingest, models, nowcast, reconcile, rent, sedlabanki, topdown

FX_COMPS = None


def _fx_comps():
    global FX_COMPS
    if FX_COMPS is None:
        FX_COMPS = [c for cs in blocks.fx_components().values() for c in cs]
    return FX_COMPS


def run_backtest(start="2025-01", max_h=12, use_reconcile=True):
    """Full-stack headline forecasts at h=1..max_h for each jump-off.

    Returns a long DataFrame: jump_off, horizon, target, actual, model,
    seasonal_naive. (RW-on-y/y is added at horizon 12 in yoy space by the caller.)
    """
    spl = ingest.load_sub_spliced(); new = ingest.load_sub_new()
    old = ingest.load_panel_old(); head = ingest.load_headline()
    g = models.build_component_history(spl, old, new)
    fx = sedlabanki.fx_mm(); hms_all = rent.hms_rent_mm()
    mm_pub = head[("CPI", "change_M")]

    jump_offs = [t for t in g.index if t >= pd.Period(start, "M") and t < g.index.max()]
    rows = []
    for jo in jump_offs:
        g_tr = g[g.index <= jo]
        w_prev = new[(new.manudur == jo) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
        if len(w_prev) < len(models.COMPONENTS):
            continue
        fits = {c: models.fit_component(g_tr[c], c) for c in models.COMPONENTS}
        fc = models.forecast_components(fits, jo, max_h, hms_rent_mm=hms_all[hms_all.index <= jo],
                                        sub_spliced=spl, comp_history=g_tr, fx_mm=fx)
        cal = nowcast.calibrate_fuel(train_end=jo)
        try:
            fc = nowcast.apply_observables(fc, nowcast.fuel_nowcast(jo + 1, cal))
        except ValueError:
            pass
        head_mm, _, _ = models.aggregate_bottom_up(fc, w_prev)

        if use_reconcile:
            contrib, _ = reconcile.contributions_from_forecast(fc, w_prev)
            tf = topdown.fit(mm_pub[mm_pub.index <= jo])
            td = topdown.forecast(tf, max_h)
            csd1 = pd.Series({c: fits[c].resid.std() * float(w_prev.get(c, 0)) / 100 for c in fc.columns})
            head_mm, _, _ = reconcile.reconcile_path(contrib, td, csd1,
                                                     topdown.error_sd_by_horizon(tf, max_h))

        for h, m in enumerate(head_mm.index, start=1):
            if m not in mm_pub.index:
                continue
            # seasonal-naive: the calendar-month mean m/m (2016+) aggregated is ~ the
            # published seasonal pattern; use published month-of-year mean as the floor
            sn = mm_pub[(mm_pub.index.month == m.month) & (mm_pub.index < jo)]["2016":].mean()
            rows.append({"jump_off": jo, "horizon": h, "target": m,
                         "actual": mm_pub[m], "model": head_mm[m], "seasonal_naive": sn})
    return pd.DataFrame(rows)


def rmse_by_horizon(bt: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE by horizon for model vs seasonal-naive (m/m space)."""
    out = []
    for h, d in bt.dropna().groupby("horizon"):
        row = {"horizon": h, "n": len(d)}
        for col in ["model", "seasonal_naive"]:
            row[f"RMSE_{col}"] = np.sqrt(((d[col] - d.actual) ** 2).mean())
            row[f"MAE_{col}"] = (d[col] - d.actual).abs().mean()
        out.append(row)
    return pd.DataFrame(out).set_index("horizon")


def yoy_backtest(bt: pd.DataFrame, head: pd.DataFrame, max_h=12) -> pd.DataFrame:
    """h=max_h y/y: model (cumulated m/m) vs RW-on-y/y, per jump-off."""
    idx = head[("CPI", "index")]; yy = head[("CPI", "change_A")]
    rows = []
    for jo, d in bt.groupby("jump_off"):
        d = d[d.horizon <= max_h].sort_values("horizon")
        if len(d) < max_h:
            continue
        tm = jo + max_h
        if tm not in yy.index:
            continue
        model_yoy = ((1 + d.model.values / 100).prod() - 1) * 100
        rows.append({"jump_off": jo, "actual_yoy": yy[tm], "model_yoy": model_yoy, "rw_yoy": yy[jo]})
    return pd.DataFrame(rows)


def h1_block_error(start="2025-02") -> pd.DataFrame:
    """h=1 contribution-error by driver block: which block drives the miss."""
    spl = ingest.load_sub_spliced(); new = ingest.load_sub_new()
    old = ingest.load_panel_old(); head = ingest.load_headline()
    g = models.build_component_history(spl, old, new)
    fx = sedlabanki.fx_mm(); hms_all = rent.hms_rent_mm()

    bmap = blocks.load_block_map()
    comp_block = {}
    for b, cfg in bmap["blocks"].items():
        for c in cfg["components"]:
            comp_block[c] = b
    for c in models.COMPONENTS:
        comp_block.setdefault(c, "other/generic")
    comp_block[models.RENT_CODE] = "imputed_rent"
    for c in ("CP0722",):
        comp_block[c] = "fuel_observed"

    rows = []
    for jo in [t for t in g.index if t >= pd.Period(start, "M") and t < g.index.max()]:
        g_tr = g[g.index <= jo]; m = jo + 1
        w_prev = new[(new.manudur == jo) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
        act = new[(new.manudur == m) & new.code.isin(models.COMPONENTS)].set_index("code").manadarbreyting
        if len(w_prev) < len(models.COMPONENTS) or len(act) < 5:
            continue
        fits = {c: models.fit_component(g_tr[c], c) for c in models.COMPONENTS}
        fc = models.forecast_components(fits, jo, 1, hms_rent_mm=hms_all[hms_all.index <= jo],
                                        sub_spliced=spl, comp_history=g_tr, fx_mm=fx)
        cal = nowcast.calibrate_fuel(train_end=jo)
        try:
            fc = nowcast.apply_observables(fc, nowcast.fuel_nowcast(m, cal))
        except ValueError:
            pass
        for c in models.COMPONENTS:
            if c in act.index:
                err = float(w_prev.get(c, 0)) / 100 * (fc.loc[m, c] - act[c])
                rows.append({"jump_off": jo, "block": comp_block[c], "contrib_error": err})
    df = pd.DataFrame(rows)
    return (df.groupby("block").contrib_error
            .agg(rmse=lambda s: np.sqrt((s ** 2).mean()), mae=lambda s: s.abs().mean(),
                 bias="mean").sort_values("rmse", ascending=False))
