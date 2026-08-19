"""Loaders that turn PxWeb tables into tidy, analysis-ready DataFrames.

Column names follow the Icelandic source terms (visitala, vaegi, ahrif)
per the project conventions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import px_client

INTERIM_DIR = px_client.REPO_ROOT / "data" / "interim"

# Discovered via px_client.discover_tables(); asserted in load-time checks below.
TABLES = {
    "headline": "1_vnv/1_vnv/VIS01000.px",
    "verdtrygging": "1_vnv/1_vnv/VIS01004.px",
    "sub_new": "1_vnv/2_undirvisitolur/VIS01300.px",        # COICOP2018, 2025M01-
    "weights_new": "1_vnv/2_undirvisitolur/VIS01306.px",    # base weights Dec2024/Dec2025
    "sub_spliced": "1_vnv/2_undirvisitolur/VIS01308.px",    # COICOP2018-coded history 1997-
    "sub_old": "1_vnv/4_eldraefni/VIS01304.px",              # old classification 2008-2025
    "panel_old": "1_vnv/4_eldraefni/VIS01301.px",            # old cls: index/vaegi/ahrif 2002-2025
    "weights_old": "1_vnv/4_eldraefni/VIS01305.px",          # base weights 1992-2024
}


def _month_index(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.str.replace("M", "-"), freq="M")


def load_headline(use_cache: bool = True) -> pd.DataFrame:
    """Headline VNV and VNV an husnaedis: index level + published changes.

    Returns a frame indexed by month with columns like
    (CPI, index), (CPI, M_rate), (CPILH, index), ...
    """
    df = px_client.fetch_table(TABLES["headline"], use_cache=use_cache)
    df["manudur"] = _month_index(df["Mánuður"])
    wide = df.pivot_table(
        index="manudur", columns=["Vísitala", "Liður"], values="value", aggfunc="first"
    )
    return wide.sort_index()


def load_verdtrygging(use_cache: bool = True) -> pd.DataFrame:
    """Visitala neysluverds til verdtryggingar (applies to indexation in t+2)."""
    df = px_client.fetch_table(TABLES["verdtrygging"], use_cache=use_cache)
    df["manudur"] = _month_index(df["Mánuður"])
    wide = df.pivot_table(index="manudur", columns="Vísitala", values="value", aggfunc="first")
    return wide.sort_index()


def load_sub_new(use_cache: bool = True) -> pd.DataFrame:
    """COICOP2018 subindex panel (2025M01-): visitala, vaegi, manadarbreyting, ahrif.

    Tidy: one row per (manudur, code) with the four measures as columns.
    """
    df = px_client.fetch_table(TABLES["sub_new"], use_cache=use_cache)
    df["manudur"] = _month_index(df["Mánuður"])
    wide = df.pivot_table(
        index=["manudur", "Undirvísitala", "Undirvísitala_text"],
        columns="Liður",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "Undirvísitala": "code",
            "Undirvísitala_text": "heiti",
            "index": "visitala",
            "breakdown": "vaegi",
            "change_M": "manadarbreyting",
            "effect": "ahrif",
        }
    )
    return wide.sort_values(["manudur", "code"]).reset_index(drop=True)


def load_panel_old(use_cache: bool = True) -> pd.DataFrame:
    """Old-classification panel 2002M08-2025M12 with the same measures."""
    df = px_client.fetch_table(TABLES["panel_old"], use_cache=use_cache)
    df["manudur"] = _month_index(df["Mánuður"])
    wide = df.pivot_table(
        index=["manudur", "Undirvísitala", "Undirvísitala_text"],
        columns="Liður",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "Undirvísitala": "code",
            "Undirvísitala_text": "heiti",
            "index_B1997": "visitala_1997",
            "index_B2008": "visitala_2008",
            "breakdown": "vaegi",
            "change_M": "manadarbreyting",
            "effect": "ahrif",
        }
    )
    return wide.sort_values(["manudur", "code"]).reset_index(drop=True)


def load_sub_spliced(use_cache: bool = True) -> pd.DataFrame:
    """Long subindex history (1997M03-) under COICOP2018 codes, spliced by Hagstofa."""
    df = px_client.fetch_table(TABLES["sub_spliced"], use_cache=use_cache)
    code_col = next(c for c in df.columns if c.startswith("COICOP"))
    df["manudur"] = _month_index(df["Mánuður"])
    out = df.rename(columns={code_col: "code", code_col + "_text": "heiti", "value": "visitala"})
    return out[["manudur", "code", "heiti", "visitala"]].sort_values(["manudur", "code"]).reset_index(drop=True)


def load_weights_new(use_cache: bool = True) -> pd.DataFrame:
    """Base weights for COICOP2018 subindices (Dec 2024 and Dec 2025 baskets)."""
    df = px_client.fetch_table(TABLES["weights_new"], use_cache=use_cache)
    wide = df.pivot_table(
        index=["Undirvísitala", "Undirvísitala_text"], columns="Tími", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns={"Undirvísitala": "code", "Undirvísitala_text": "heiti"})


def load_weights_old(use_cache: bool = True) -> pd.DataFrame:
    """Annual base weights 1992-2024, old classification."""
    df = px_client.fetch_table(TABLES["weights_old"], use_cache=use_cache)
    wide = df.pivot_table(
        index=["Undirvísitala", "Undirvísitala_text"], columns="Tími", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns={"Undirvísitala": "code", "Undirvísitala_text": "heiti"})
