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

BLOCK_ORDER = ["imputed_rent", "food", "imported_goods", "domestic_services",
               "fuel", "airfares", "other"]

# How each component's forward path is derived. (label, description) — Icelandic
# then English. Specific components first; the rest fall back by driver block.
_METHOD_BY_CODE = {
    "CP042": ("HMS leiguígildi + þrautseigja",
              "EWMA (helmingunartími 6 mán.) af eigin m/m + 30% vog á nýja "
              "leigusamninga HMS (tafir 1–3). Aðeins eftir júní 2024 (aðferðabrot "
              "í leiguígildi). Slær RW og AR(1) í prófun. / EWMA persistence of "
              "own m/m + a shrunk tilt on HMS new-contract rents (lags 1–3); "
              "post-June-2024 sample only; beats RW and AR(1) out of sample."),
    "CP0722": ("Mælt: dæluverð (Gasvaktin)",
               "Núspáin er MÆLD: meðaldæluverð yfir söfnunarglugga (1.–15.) frá "
               "Gasvaktin, kvarðað á birta undirvísitölu (β≈1,24). / Nowcast is "
               "OBSERVED: collection-window pump prices calibrated to the "
               "published subindex."),
    "CP073": ("Flugfargjöld — í söfnun",
              "Söfnun KEF-fargjalda frá Icelandair hafin; kvörðun bíður ≥2 "
              "söfnunarglugga. Þangað til: árstíðar + AR líkan. / Icelandair KEF "
              "fares collecting; calibration pending — generic seasonal+AR meanwhile."),
    "CP041": ("Greidd húsaleiga — árstíðar + AR",
              "Árstíðarstuðlar + AR(1). Fylgir sömu leigu og CP042 en með "
              "samningatöf. / Seasonal + AR(1); tracks the same rents as CP042 "
              "with a contract lag."),
    "CP045": ("Rafmagn/hiti — árstíðar + AR",
              "Árstíðar + AR(1). Að mestu opinberar gjaldskrár; skref má færa inn "
              "um fjármálaalmanak. / Seasonal + AR(1); largely administered "
              "tariffs, dated steps can enter via the fiscal calendar."),
}
_METHOD_BY_BLOCK = {
    "food": ("Árstíðar + AR + gengisáhrif",
             "Árstíðarstuðlar + AR(1) með rýrðri gengisvog (gengisvísitala, tafir "
             "0–3; 12-mán. gegnumstreymi ≈0,2). / Seasonal + AR(1) with a shrunk "
             "FX pass-through tilt (~0.2 over 12 months)."),
    "imported_goods": ("Árstíðar + AR + gengisáhrif",
                       "Árstíðar + AR(1) + gengis-gegnumstreymi (12-mán. ≈0,19 "
                       "föt, ≈0,43 bílar — í takt við væntingar áætlunar). / "
                       "Seasonal + AR(1) + FX pass-through (~0.19 clothing, ~0.43 "
                       "vehicles, in the plan's expected range)."),
    "domestic_services": ("Árstíðar + AR + kjarasamningar",
                          "Árstíðar + AR(1); dagsett kjarasamningaskref úr "
                          "wage_calendar.yaml (laun/þjónustuverð fylgni er ~0, því "
                          "almanak fremur en aðhvarf). / Seasonal + AR(1) with "
                          "dated kjarasamningar steps (the wage→price regression "
                          "is ~0, so a calendar, not a regression)."),
    "other": ("Árstíðar + AR(1)",
              "Árstíðarstuðlar (frá 2016) + AR(1) á árstíðarleiðréttri m/m. "
              "Grunnlíkan þar sem enginn sterkur drifkraftur greinist. / Seasonal "
              "factors (2016–) + AR(1) on the deseasonalized m/m — the baseline "
              "where no strong driver is identified."),
}


def component_method(code: str) -> tuple[str, str]:
    """(label, description) of how this component's forward path is derived."""
    if code in _METHOD_BY_CODE:
        return _METHOD_BY_CODE[code]
    return _METHOD_BY_BLOCK.get(_block_of(code), _METHOD_BY_BLOCK["other"])


def _component_levels() -> pd.DataFrame:
    """Monthly published subindex LEVELS per component (spliced long history,
    filled with the new COICOP2018 panel where the spliced series lacks a code)."""
    spl = ingest.load_sub_spliced(); new = ingest.load_sub_new()
    lvl = spl.pivot_table(index="manudur", columns="code", values="visitala")
    new_lvl = new.pivot_table(index="manudur", columns="code", values="visitala")
    for c in models.COMPONENTS:
        if c not in lvl.columns or lvl[c].dropna().empty:
            if c in new_lvl.columns:
                lvl[c] = new_lvl[c]
    return lvl.sort_index()


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

    # per-component detail: history + forecast index trend, method, contribution
    details = component_details(fc, contribs, w0, last_m)

    return dict(last_m=last_m, nowcast_month=last_m + 1, nowcast_mm=nowcast_mm,
                yy_12m=yy_12m, band90=(yy_12m - 1.64 * cum_sd, yy_12m + 1.64 * cum_sd),
                yy_path=(path_idx / idx_hist.reindex(path_idx.index - 12).to_numpy() - 1) * 100,
                path_idx=path_idx, rec_sd=rec_sd,
                waterfall=waterfall, verdtrygging=vt, model_vs_breakeven=mvb,
                path_mm=rec_head, fc=fc, contribs=contribs, details=details)


def component_details(fc: pd.DataFrame, contribs: pd.DataFrame, w0: pd.Series,
                      last_m, hist_months: int = 42) -> list[dict]:
    """One record per component for the report's per-underlying section.

    Each has: code, heiti, block, weight, 12-month contribution, method label +
    description, and a level trend (recent history + forecast projection).
    Ordered by block, then by absolute 12-month contribution within block.
    """
    levels = _component_levels()
    new = ingest.load_sub_new()
    heiti = new[new.manudur == last_m].set_index("code")["heiti"]
    contrib12 = contribs.sum()

    out = []
    for c in models.COMPONENTS:
        hist = levels[c].dropna() if c in levels.columns else pd.Series(dtype=float)
        hist = hist[hist.index > last_m - hist_months]
        anchor = hist.iloc[-1] if len(hist) else 100.0
        fpath = anchor * (1 + fc[c] / 100).cumprod()
        label, desc = component_method(c)
        out.append({
            "code": c, "heiti": heiti.get(c, c), "block": _block_of(c),
            "weight": float(w0.get(c, 0)), "contrib12": float(contrib12.get(c, 0)),
            "mm_next": float(fc.iloc[0][c]), "method_label": label, "method_desc": desc,
            "hist_index": hist, "fcst_index": fpath,
        })
    order = {b: i for i, b in enumerate(BLOCK_ORDER)}
    out.sort(key=lambda r: (order.get(r["block"], 99), -abs(r["contrib12"])))
    return out


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
    lines += ["", "## Módel vs verðbólguálag / Model vs breakeven", ""]
    if mvb is None:
        lines.append("_Breakeven slot óupppfyllt — keyra `lanamal.update_breakeven_slot()`._")
    else:
        lines.append(f"- Módel 12-mán. verðbólga / model 12-month: {mvb['model_yoy']:.2f}%  ")
        lines.append(f"- Markaðs-verðbólguálag (RIKB−RIKS), "
                     f"stysta {mvb['breakeven_horizon_yrs']:.0f}á / shortest breakeven: "
                     f"{mvb['breakeven']:.2f}%  ")
        lines.append(f"- Fleygur / wedge (álag − módel): {mvb['implied_wedge']:+.2f}pp "
                     "_(ólíkir sjóndeildarhringir; álag ber áhættu- og skortsálag / "
                     "horizons differ; breakeven carries risk & scarcity premia)_")
        lines.append("")
        lines.append("| Sjóndeild. / Horizon (yr) | Verðbólguálag / Breakeven (%) |")
        lines.append("|---|---|")
        for r in mvb["curve"]:
            lines.append(f"| {r['horizon_yrs']:.0f} | {r['breakeven_pct']:.2f} |")
    return "\n".join(lines)
