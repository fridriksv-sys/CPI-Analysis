"""MinT reconciliation of the bottom-up and top-down paths (Phase 6).

Setting (Wickramasuriya/Athanasopoulos/Hyndman): a 2-level hierarchy — headline
total over n component CONTRIBUTIONS c_i = (weight_i/100) * mm_i. In contribution
space the aggregation is a plain sum (headline m/m = Σ c_i), so the summing matrix
is S = [1ᵀ; Iₙ] and MinT is exact and simple.

Base (incoherent) forecasts per horizon h:
  aggregate  â_h  = top-down headline model
  bottom     ĉ_h  = bottom-up contributions (component model × price-updated weight)
These generally do not satisfy Σĉ = â. MinT finds the coherent set closest (in a
W⁻¹ trace-minimizing sense) to the base forecasts:
  b̃ = (SᵀW⁻¹S)⁻¹ SᵀW⁻¹ [â; ĉ],   reconciled headline = Σ b̃.

W is diagonal (WLS-variance MinT — the stable choice for short samples), with
HORIZON-SPECIFIC variances: the top-down error grows slowly (mean-reverting AR)
while bottom-up contribution errors compound. So at h=1 the detailed bottom-up is
preserved; by h=12 the reconciliation pulls the path toward the top-down — the
behaviour the plan expects, not a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reconcile_horizon(agg: float, contrib: np.ndarray,
                      sd_agg: float, sd_contrib: np.ndarray) -> np.ndarray:
    """MinT-reconciled bottom contributions for one horizon (diagonal W).

    Solves (SᵀW⁻¹S) b = SᵀW⁻¹ y in closed form for S = [1ᵀ; I].
    Returns the reconciled contributions b (sum = reconciled aggregate).
    """
    n = len(contrib)
    a0 = 1.0 / sd_agg ** 2
    d = 1.0 / np.asarray(sd_contrib, float) ** 2
    M = np.diag(d) + a0 * np.ones((n, n))
    rhs = a0 * agg * np.ones(n) + d * np.asarray(contrib, float)
    return np.linalg.solve(M, rhs)


def reconcile_path(
    contrib_fc: pd.DataFrame,       # rows=months, cols=components (contributions, pp)
    topdown_mm: pd.Series,          # aggregate headline m/m per month
    contrib_sd1: pd.Series,         # h=1 error SD per component contribution
    topdown_sd: pd.Series,          # top-down error SD per horizon
    contrib_compounding: float = 0.5,   # bottom-up SD grows ~ h**this
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Reconcile the whole path.

    Returns (reconciled headline m/m, reconciled contributions, reconciled
    headline error SD per horizon). The SD comes from the MinT reconciled
    covariance (SᵀW⁻¹S)⁻¹ — the aggregate variance is 1ᵀ(SᵀW⁻¹S)⁻¹1 — so the
    fan chart reflects the reconciliation, not a naive sum of component errors.
    """
    months = contrib_fc.index
    comps = list(contrib_fc.columns)
    sd1 = contrib_sd1.reindex(comps).fillna(contrib_sd1.median()).to_numpy()

    rec_rows, heads, head_sd = [], [], []
    for h, m in enumerate(months, start=1):
        sd_c = sd1 * (h ** contrib_compounding)
        a0 = 1.0 / float(topdown_sd.loc[m]) ** 2
        d = 1.0 / sd_c ** 2
        M = np.diag(d) + a0 * np.ones((len(comps), len(comps)))
        rhs = a0 * float(topdown_mm.loc[m]) * np.ones(len(comps)) + d * contrib_fc.loc[m].to_numpy()
        Minv = np.linalg.inv(M)
        b = Minv @ rhs
        rec_rows.append(b)
        heads.append(b.sum())
        head_sd.append(np.sqrt(np.ones(len(comps)) @ Minv @ np.ones(len(comps))))
    rec = pd.DataFrame(rec_rows, index=months, columns=comps)
    return (pd.Series(heads, index=months, name="reconciled_mm"), rec,
            pd.Series(head_sd, index=months, name="reconciled_sd"))


def contributions_from_forecast(fcst_mm: pd.DataFrame, w0: pd.Series):
    """Turn component m/m forecasts into contributions with the price-updated
    weight path (same engine as Phase 2). Returns (contrib_df, weight_path)."""
    from .reconstruct import price_update_weights
    w = w0 / w0.sum() * 100
    contribs, weights = [], []
    for _, row in fcst_mm.iterrows():
        contribs.append(w * row / 100)
        weights.append(w.copy())
        w = price_update_weights(w, row)
    return (pd.DataFrame(contribs, index=fcst_mm.index),
            pd.DataFrame(weights, index=fcst_mm.index))
