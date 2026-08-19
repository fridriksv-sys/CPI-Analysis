"""Component forecast models (v0 baseline).

Deliberately simple and fully inspectable: each component's m/m change is
modeled as estimated seasonal factors plus an AR(1) on the deseasonalized
series. The one exception is reiknud husaleiga (CP042), where the June-2024
methodology break (user cost -> rental equivalence) makes pre-break history
uninformative; it gets a persistence model on post-break data only.

This is the Phase 5 'v0' layer of PLAN_1.md: no scraped nowcast inputs yet
(airfares, fuel, groceries), no fiscal/wage calendars, no MinT reconciliation.
Those raise accuracy at h=1-3; the skeleton here is the aggregation-correct
bottom-up path they plug into.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Forecast components: 13 COICOP2018 divisions with housing (CP04) split into
# its groups so reiknud husaleiga is modeled on its own regime, and transport
# (CP07) split so eldsneyti (CP0722) can take the observable fuel nowcast.
COMPONENTS = [
    "CP01", "CP02", "CP03",
    "CP041", "CP042", "CP043", "CP044", "CP045",
    "CP05", "CP06",
    "CP071", "CP0721", "CP0722", "CP0723", "CP0724", "CP073", "CP074",
    "CP08", "CP09", "CP10", "CP11", "CP12", "CP13",
]

# Components whose m/m can be OBSERVED (scraped) rather than modeled by mid-month.
# CP0722 fuel is live; CP073 airfares collect now, override activates once
# calibrated (>= 2 collection windows).
OBSERVABLE = {"CP0722", "CP073"}

RENT_CODE = "CP042"
RENT_BREAK = pd.Period("2024-07", "M")  # first m/m fully under rental equivalence
DEFAULT_SAMPLE_START = pd.Period("2016-01", "M")

# Divisions restructured in COICOP2018 have no spliced VIS01308 history at
# division level. For PARAMETER FITTING ONLY (seasonality, persistence) we use
# the nearest old-classification division from VIS01301 as proxy history.
# Old IS12 (misc goods/services) covers what COICOP2018 splits into CP12+CP13.
# These proxies never enter index construction; v0 approximation, flagged for
# Phase 5 refinement.
OLD_PROXY = {
    "CP043": "IS043",  # vidhald husnaedis
    "CP074": "IS07",   # voruflutningar (0.13% weight; division proxy is fine)
    "CP08": "IS08",    # fjarskipti (old) ~ upplysingar og fjarskipti (new)
    "CP09": "IS09",    # tomstundir ~ afthreying/ithrottir/menning
    "CP10": "IS10",    # menntun
    "CP12": "IS12",
    "CP13": "IS12",
}


def build_component_history(
    spl: pd.DataFrame, panel_old: pd.DataFrame, sub_new: pd.DataFrame
) -> pd.DataFrame:
    """Assemble one m/m (%) history per forecast component.

    Priority per component: spliced COICOP2018 levels (VIS01308) -> old-
    classification proxy m/m (VIS01301) -> always overridden from 2025-01 by
    the new panel's published m/m changes (VIS01300, exact to 2dp).
    """
    lvl = spl[spl.code.isin(COMPONENTS)].pivot_table(
        index="manudur", columns="code", values="visitala"
    )
    g = lvl.pct_change() * 100

    old_mm = panel_old.pivot_table(index="manudur", columns="code", values="manadarbreyting")
    for cp, is_code in OLD_PROXY.items():
        if cp not in g.columns:
            g[cp] = old_mm[is_code]

    new_mm = sub_new.pivot_table(index="manudur", columns="code", values="manadarbreyting")
    new_part = new_mm.reindex(columns=COMPONENTS).dropna(how="all")
    g = g.reindex(columns=COMPONENTS)
    g.loc[g.index >= new_part.index.min(), :] = new_part
    return g.sort_index()


@dataclass
class ComponentFit:
    code: str
    seasonal: pd.Series          # 12 monthly factors (pp)
    phi: float                   # AR(1) on deseasonalized m/m
    resid: pd.Series             # in-sample residuals (for bootstrap)
    last_deseason: float         # jump-off deseasonalized m/m
    n_obs: int
    note: str = ""


def _fit_seasonal_ar(g: pd.Series, min_years: int = 5) -> tuple[pd.Series, float, pd.Series]:
    """Monthly-mean seasonal factors + AR(1) via OLS on the deseasonalized series."""
    month = g.index.month
    seasonal = g.groupby(month).mean()
    de = g - seasonal.reindex(month).to_numpy()
    x, y = de.shift(1).dropna(), de.iloc[1:]
    x, y = x.align(y, join="inner")
    phi = 0.0
    if len(x) >= 24 and x.var() > 0:
        phi = float(np.clip((x * y).sum() / (x * x).sum(), -0.5, 0.9))
    resid = y - phi * x
    return seasonal, phi, resid


def fit_component(g_mm: pd.Series, code: str) -> ComponentFit:
    """Fit one component's m/m model on the appropriate sample."""
    g = g_mm.dropna()
    if code == RENT_CODE:
        # Post-break only: rents under rental equivalence are highly persistent
        # and carry no useful seasonal signal from the old user-cost regime.
        g = g[g.index >= RENT_BREAK]
        mu = g.tail(12).mean()
        phi = 0.7  # persistence prior: stock rents adjust slowly (PLAN_1 Phase 4)
        seasonal = pd.Series(mu, index=range(1, 13))
        de = g - mu
        resid = (de - phi * de.shift(1)).dropna()
        return ComponentFit(code, seasonal, phi, resid, float(de.iloc[-1]), len(g),
                            note=f"post-{RENT_BREAK} sample only (methodology break)")
    g = g[g.index >= DEFAULT_SAMPLE_START]
    seasonal, phi, resid = _fit_seasonal_ar(g)
    de_last = float(g.iloc[-1] - seasonal[g.index[-1].month])
    return ComponentFit(code, seasonal, phi, resid, de_last, len(g))


def forecast_component(fit: ComponentFit, start: pd.Period, horizons: int = 12) -> pd.Series:
    """Point forecast of m/m (%) for months start+1 .. start+horizons."""
    months = pd.period_range(start + 1, periods=horizons, freq="M")
    de = fit.last_deseason
    out = []
    for m in months:
        de = fit.phi * de
        out.append(fit.seasonal[m.month] + de)
    return pd.Series(out, index=months)


def forecast_components(
    fits: dict[str, "ComponentFit"], jump_off, horizons: int = 12,
    hms_rent_mm=None, sub_spliced=None,
):
    """Forecast every component m/m, routing CP042 through the Phase 4 rent model.

    If hms_rent_mm + sub_spliced are supplied, the reiknuð húsaleiga column is
    replaced by rent.forecast_cp042 (EWMA persistence + HMS new-contract tilt);
    otherwise CP042 falls back to its generic seasonal/persistence fit.
    """
    fc = pd.DataFrame({c: forecast_component(fits[c], jump_off, horizons) for c in COMPONENTS})
    if hms_rent_mm is not None and sub_spliced is not None and RENT_CODE in fc.columns:
        from .rent import forecast_cp042, load_cp042_history
        cp042_hist = load_cp042_history(sub_spliced)
        fc[RENT_CODE] = forecast_cp042(cp042_hist, hms_rent_mm, jump_off, horizons)
    return fc


def aggregate_bottom_up(
    fcst_mm: pd.DataFrame, w0: pd.Series, price_update=None
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Aggregate component m/m forecasts into the headline path.

    fcst_mm: rows = months, columns = component codes (m/m %).
    w0: jump-off weight shares (sum 100), from the latest published Vaegi.
    Returns (headline m/m series, contributions frame, weight-path frame).
    Weights are price-updated through the horizon with the same rule the
    Phase 2 reconstruction verified against published Vaegi.
    """
    from .reconstruct import price_update_weights
    w = w0 / w0.sum() * 100
    heads, contribs, weights = [], [], []
    for m, row in fcst_mm.iterrows():
        contrib = w * row / 100
        heads.append(contrib.sum())
        contribs.append(contrib)
        weights.append(w.copy())
        w = price_update_weights(w, row)
    months = fcst_mm.index
    return (
        pd.Series(heads, index=months, name="headline_mm"),
        pd.DataFrame(contribs, index=months),
        pd.DataFrame(weights, index=months),
    )


def bootstrap_paths(
    fits: dict[str, ComponentFit], fcst_mm: pd.DataFrame, w0: pd.Series,
    n_sims: int = 2000, seed: int = 42,
) -> pd.DataFrame:
    """Simulate headline m/m paths by resampling joint residual months.

    To preserve cross-component correlation, one historical month is drawn per
    simulated month and every component's residual is taken from that same
    month (components missing that month draw independently).
    """
    rng = np.random.default_rng(seed)
    codes = list(fcst_mm.columns)
    resid = pd.DataFrame({c: fits[c].resid for c in codes})
    resid = resid.dropna(how="all")
    filled = resid.apply(lambda col: col.fillna(pd.Series(
        rng.choice(col.dropna().to_numpy(), size=len(col)), index=col.index)))
    months = fcst_mm.index
    w_base = (w0 / w0.sum() * 100)

    sims = np.empty((n_sims, len(months)))
    pool = filled.to_numpy()
    for s in range(n_sims):
        draw = pool[rng.integers(0, len(pool), size=len(months))]
        shocked = fcst_mm.to_numpy() + draw
        w = w_base.to_numpy().copy()
        for t in range(len(months)):
            g = shocked[t]
            sims[s, t] = float((w * g).sum() / 100)
            w = w * (1 + g / 100)
            w = w / w.sum() * 100
    return pd.DataFrame(sims, columns=months)
