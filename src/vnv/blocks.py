"""Driver-based ARDL blocks (Phase 5).

Two findings from the data (scripts/explore_blocks.py) shape this module:

1. FX pass-through into imported goods and food is REAL and lands in the plan's
   expected 0.2-0.4 range over 12 months (CP071 vehicles ~0.43, CP03 clothing
   ~0.19, CP01 food ~0.20), but monthly R^2 is low (0.08-0.23). So FX enters as
   a SHRUNK tilt on top of each component's seasonal/AR fit, not as a standalone
   regression that would chase noise.

2. The wage->services regression is near-useless (R^2~0.02): services adjust to
   wages in discrete kjarasamningar steps with long inertia, not month to month.
   Per PLAN_1.md §5 those steps come from a dated calendar (wage_calendar.yaml),
   applied to the domestic-services block in the forecast PATH, not discovered by
   a regression.

Observability at nowcast time (~day 15): the FX sample is the mid-collection-
window value, so FX at lag 0 is observed; FAO food publishes early in the month
(lag 0 observed); wages publish with a ~1-month lag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import ingest, sedlabanki
from .px_client import REPO_ROOT

CONFIG = REPO_ROOT / "config"
FX_LAGS = (0, 1, 2, 3)
FX_TILT = 0.35          # shrinkage weight on the FX pass-through component
SAMPLE_START = pd.Period("2020-01", "M")  # FX history starts 2020


def load_block_map() -> dict:
    return yaml.safe_load((CONFIG / "blocks.yaml").read_text(encoding="utf-8"))


def fx_components() -> dict[str, list[str]]:
    """{block_name: [component codes]} for FX-driven blocks (food + imported)."""
    cfg = load_block_map()["blocks"]
    return {b: cfg[b]["components"] for b in ("food", "imported_goods") if b in cfg}


def _deseasonalize(y: pd.Series) -> tuple[pd.Series, pd.Series]:
    seas = y.groupby(y.index.month).mean()
    de = y - seas.reindex(y.index.month).to_numpy()
    return de, seas


def fit_fx_passthrough(comp_mm: pd.Series, fx_mm: pd.Series,
                       train_end: pd.Period | None = None) -> dict | None:
    """Distributed-lag FX pass-through on the deseasonalized component m/m.

    Returns the lag coefficients and seasonal factors, or None if the sample is
    too short. The cumulative pass-through is sum(coefs).
    """
    d = pd.concat([comp_mm.rename("y")]
                  + [fx_mm.shift(k).rename(f"d{k}") for k in FX_LAGS], axis=1)
    d = d[d.index >= SAMPLE_START].dropna()
    if train_end is not None:
        d = d[d.index <= train_end]
    if len(d) < 30:
        return None
    de, seas = _deseasonalize(d.y)
    X = np.column_stack([np.ones(len(d))] + [d[f"d{k}"].values for k in FX_LAGS])
    beta, *_ = np.linalg.lstsq(X, de.values, rcond=None)
    return {"const": beta[0], "coefs": beta[1:], "seasonal": seas,
            "passthrough": float(beta[1:].sum()), "n": len(d)}


def fx_tilt_forecast(fit: dict, fx_mm: pd.Series, month: pd.Period) -> float | None:
    """Deseasonalized FX-driven m/m for `month` (add the seasonal back to use)."""
    if fit is None:
        return None
    lags = [fx_mm.get(month - k, np.nan) for k in FX_LAGS]
    if any(np.isnan(v) for v in lags):
        return None
    return float(fit["const"] + np.dot(fit["coefs"], lags))


# ---------------------------------------------------------------- wage calendar
def load_wage_calendar() -> list[dict]:
    path = CONFIG / "wage_calendar.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def wage_step(month: pd.Period, component: str) -> float:
    """Additive m/m (pp) from dated kjarasamningar steps hitting `component`."""
    total = 0.0
    for step in load_wage_calendar():
        if str(step.get("date")) == str(month) and component in step.get("affects", []):
            total += float(step.get("mm_pp") or 0.0)
    return total


def domestic_service_components() -> list[str]:
    cfg = load_block_map()["blocks"].get("domestic_services", {})
    return cfg.get("components", [])
