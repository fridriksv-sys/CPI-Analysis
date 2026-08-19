"""Reconstruction of the published headline VNV from component data.

Two independent reconstructions, matching the two publication eras:

1. Old classification (VIS01301), 2003-2025: the panel publishes a monthly
   price-updated weight (vaegi) per subindex. The Laspeyres identity is

       headline m/m (%)  =  sum_i  vaegi_{i,t-1} * change_{i,t} / 100

   where vaegi_{i,t-1} is the component's share of basket value at t-1 prices.

2. COICOP2018 (VIS01300 + VIS01306), 2026-: the index is fixed-base Laspeyres
   on the December 2025 basket, so levels aggregate directly:

       headline_t = sum_i w_i(Dec) * I_{i,t} / 100      (I rebased Dec=100)

Precision note: Hagstofa's API stores indices at 1 decimal and weights/changes/
effects at 2 decimals. Reconstruction errors must therefore be judged against a
rounding budget, not against zero. The tolerance functions below compute that
budget explicitly so 'pass' means 'exact up to published rounding'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIV_OLD = [f"IS{i:02d}" for i in range(1, 13)]   # IS01..IS12 (IS00 = headline)
DIV_NEW = [f"CP{i:02d}" for i in range(1, 14)]   # CP01..CP13 (CP00 = headline)


def mm_from_panel(panel: pd.DataFrame, div_codes: list[str]) -> pd.DataFrame:
    """Reconstruct headline m/m from division weights and changes.

    The Laspeyres contribution of component i in month t is its basket share at
    t-1 prices times its m/m change. We obtain that share by de-updating the
    published month-t share by the component's own change:

        w_pre_i = vaegi_it / (1 + g_it/100),  renormalized to sum to 100.

    For ordinary months this equals vaegi at t-1 identically. In base-change
    months (April up to 2024, January from 2025) vaegi(t-1) belongs to the OLD
    basket while the index is computed on the NEW one; de-updating vaegi(t)
    handles those months with the same formula, no special-casing.
    """
    div = panel[panel.code.isin(div_codes)].copy().sort_values(["code", "manudur"])
    div["w_pre_raw"] = div["vaegi"] / (1 + div["manadarbreyting"] / 100)
    wsum = div.groupby("manudur")["w_pre_raw"].transform("sum")
    div["w_pre"] = div["w_pre_raw"] / wsum * 100
    div["contrib"] = div["w_pre"] * div["manadarbreyting"] / 100

    out = div.groupby("manudur").agg(
        recon_mm=("contrib", "sum"),
        sum_ahrif=("ahrif", "sum"),
        n_components=("code", "count"),
    )
    head_code = "IS00" if div_codes[0].startswith("IS") else "CP00"
    published = panel[panel.code == head_code].set_index("manudur")["manadarbreyting"]
    out["published_mm"] = published.reindex(out.index)
    out["error"] = out["recon_mm"] - out["published_mm"]
    return out.dropna(subset=["recon_mm", "published_mm"]).loc[lambda d: d.n_components == len(div_codes)]


def levels_from_mm(recon_mm: pd.Series, anchor_month, anchor_level: float) -> pd.Series:
    """Chain reconstructed m/m rates into index levels from a published anchor."""
    mm = recon_mm.sort_index()
    mm = mm[mm.index > anchor_month]
    return anchor_level * (1 + mm / 100).cumprod()


def levels_new_era(sub_new: pd.DataFrame, weights_dec: pd.Series, from_month="2026-01") -> pd.DataFrame:
    """Fixed-base Laspeyres aggregation for the COICOP2018 era (Dec 2025 = 100).

    weights_dec: December-2025 basket weights indexed by division code (sums to 100).
    Returns reconstructed headline level (Dec 2025 = 100) vs the published CP00 index.
    """
    sub = sub_new[sub_new.manudur >= from_month]
    div = sub[sub.code.isin(weights_dec.index)]
    pivot = div.pivot_table(index="manudur", columns="code", values="visitala", aggfunc="first")
    missing = set(weights_dec.index) - set(pivot.columns)
    if missing:
        raise ValueError(f"missing divisions in subindex data: {missing}")
    recon = (pivot[weights_dec.index] * weights_dec.values).sum(axis=1) / weights_dec.sum()
    published = sub[sub.code == ("CP00")].set_index("manudur")["visitala"].reindex(recon.index)
    return pd.DataFrame({"recon_level": recon, "published_level": published, "error": recon - published})


def rounding_budget_mm(vaegi: pd.Series, n_components: int) -> float:
    """Worst-case reconstruction error (pp) from published rounding alone.

    change is rounded to 2dp (+-0.005pp per component, scaled by weight) and
    vaegi to 2dp (+-0.005 weight points, scaled by typical change size ~1%).
    The published headline m/m is itself rounded to 2dp (+-0.005pp).
    Worst case adds linearly; in practice errors are much smaller (RMS-like).
    """
    change_term = (vaegi / 100 * 0.005).sum()
    weight_term = n_components * 0.00005 * 1.0  # 0.005 weight-pts x ~1% typical change
    headline_term = 0.005
    return change_term + weight_term + headline_term


def price_update_weights(w_prev: pd.Series, mm_pct: pd.Series) -> pd.Series:
    """One-step price update of weight shares given component m/m changes (%).

    w_it = w_{i,t-1} (1+g_it) / sum_j w_{j,t-1} (1+g_jt)  -- shares stay summed to 100.
    This is how a fixed-quantity (Laspeyres) basket's value shares evolve, and it
    reproduces Hagstofa's published monthly 'Vaegi %'.
    """
    grown = w_prev * (1 + mm_pct.reindex(w_prev.index) / 100)
    return grown / grown.sum() * 100
