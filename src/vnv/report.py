"""Monthly one-pager output (Phase 7).

Assembles the model's headline result into a compact report: point nowcast and
12-month path, m/m contribution waterfall BY DRIVER BLOCK, the implied
verðtrygging path (VNV in month t sets indexation in t+2), and the model-vs-
breakeven decomposition when the breakeven slot is populated. English + Icelandic
labels, per PLAN_1.md §7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import (benchmarks, blocks, ingest, models, nowcast, reconcile, rent,
               sedlabanki, topdown)


def _block_of(code: str) -> str:
    bmap = blocks.load_block_map()
    for b, cfg in bmap["blocks"].items():
        if code in cfg["components"]:
            return b
    if code == models.RENT_CODE:
        return "imputed_rent"
    if code == "CP0722":
        return "fuel"
    if code == "CP073":
        return "airfares"
    return "other"


BLOCK_LABELS = {
    "food": ("Matur & drykkur", "Food & beverages"),
    "imported_goods": ("Innfluttar vörur", "Imported goods"),
    "domestic_services": ("Innlend þjónusta", "Domestic services"),
    "imputed_rent": ("Reiknuð húsaleiga", "Imputed rent"),
    "fuel": ("Eldsneyti", "Fuel"),
    "airfares": ("Flugfargjöld", "Airfares"),
    "other": ("Annað", "Other"),
}


def build_report() -> dict:
    head = ingest.load_headline()
    spl = ingest.load_sub_spliced(); new = ingest.load_sub_new(); old = ingest.load_panel_old()
    g = models.build_component_history(spl, old, new)
    last_m = g.index.max()
    idx_hist = head[("CPI", "index")]
    w0 = new[(new.manudur == last_m) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi

    fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
    fc = models.forecast_components(fits, last_m, 12, hms_rent_mm=rent.hms_rent_mm(),
                                    sub_spliced=spl, comp_history=g, fx_mm=sedlabanki.fx_mm())
    cal = nowcast.calibrate_fuel()
    try:
        obs = nowcast.fuel_nowcast(last_m + 1, cal)
    except ValueError:
        obs = {}
    fc = nowcast.apply_observables(fc, obs)

    _, contribs, _ = models.aggregate_bottom_up(fc, w0)
    ctb, _ = reconcile.contributions_from_forecast(fc, w0)
    tf = topdown.fit(head[("CPI", "change_M")].dropna())
    td = topdown.forecast(tf, 12)
    csd1 = pd.Series({c: fits[c].resid.std() * float(w0.get(c, 0)) / 100 for c in fc.columns})
    rec_head, _, rec_sd = reconcile.reconcile_path(ctb, td, csd1, topdown.error_sd_by_horizon(tf, 12))

    # headline nowcast (h=1) and 12-month path
    nowcast_mm = rec_head.iloc[0]
    path_idx = idx_hist.iloc[-1] * (1 + rec_head / 100).cumprod()
    yy_12m = (path_idx.iloc[-1] / idx_hist[path_idx.index[-1] - 12] - 1) * 100
    cum_sd = float(np.sqrt((rec_sd ** 2).cumsum()).iloc[-1])

    # contribution waterfall by block (12-month sum)
    blk = contribs.sum().groupby(_block_of).sum().sort_values(ascending=False)
    waterfall = pd.DataFrame({
        "block": [BLOCK_LABELS.get(b, (b, b))[0] for b in blk.index],
        "block_en": [BLOCK_LABELS.get(b, (b, b))[1] for b in blk.index],
        "framlag_pp": blk.values.round(3),
    })

    # verðtrygging path (t -> t+2)
    vt = pd.DataFrame({"VNV_spa": path_idx.round(1)})
    vt.index = path_idx.index + 2
    vt.index.name = "verdtryggingarmanudur"

    # model vs breakeven (if slot populated)
    mvb = benchmarks.model_vs_breakeven(yy_12m)

    return dict(last_m=last_m, nowcast_month=last_m + 1, nowcast_mm=nowcast_mm,
                yy_12m=yy_12m, band90=(yy_12m - 1.64 * cum_sd, yy_12m + 1.64 * cum_sd),
                waterfall=waterfall, verdtrygging=vt, model_vs_breakeven=mvb,
                path_mm=rec_head)


def to_markdown(rep: dict) -> str:
    lo, hi = rep["band90"]
    lines = [
        f"# VNV spá / CPI forecast — {rep['last_m']}",
        "",
        f"**Núspá {rep['nowcast_month']} (m/m):** {rep['nowcast_mm']:+.2f}%  ",
        f"**12-mánaða verðbólga / 12-month inflation:** {rep['yy_12m']:.1f}%  "
        f"(90% bil {lo:.1f}–{hi:.1f}%)",
        "",
        "## Framlag eftir drifkrafti / Contribution by driver block (12m, pp)",
        "",
        "| Blokk / Block | Framlag / Contribution (pp) |",
        "|---|---|",
    ]
    for _, r in rep["waterfall"].iterrows():
        lines.append(f"| {r.block} / {r.block_en} | {r.framlag_pp:+.2f} |")
    lines += ["", "## Verðtrygging (VNV í t → verðtrygging í t+2)", ""]
    vt = rep["verdtrygging"]
    lines.append("| Mánuður | VNV (spá) |")
    lines.append("|---|---|")
    for m, r in vt.head(4).iterrows():
        lines.append(f"| {m} | {r.VNV_spa:.1f} |")
    lines.append("| … | … |")
    mvb = rep["model_vs_breakeven"]
    lines += ["", "## Módel vs breakeven / Model vs breakeven", ""]
    if mvb is None:
        lines.append("_Breakeven slot óupppfyllt — sjá data/benchmarks/breakeven.csv "
                     "(needs LSEG/Keldan feed)._")
    else:
        lines.append(f"- Módel y/y: {mvb['model_yoy']:.2f}%  ")
        lines.append(f"- Breakeven ({mvb['horizon_yrs']}y): {mvb['breakeven']:.2f}%  ")
        lines.append(f"- Óbeint álag / implied premium: {mvb['implied_premium']:+.2f}pp")
    return "\n".join(lines)
