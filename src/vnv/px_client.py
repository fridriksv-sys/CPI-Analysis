"""PxWeb API client for Hagstofa Islands (Statistics Iceland).

All responses are cached to data/raw/ so notebooks run offline after the
first fetch. No table IDs are hardcoded here; discovery happens via the API
directory tree (see discover_tables / config/px_tables.yaml).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://px.hagstofa.is/pxis/api/v1/is/Efnahagur/visitolur"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Hagstofa's PxWeb instance rejects queries above ~100k cells; chunk under it.
MAX_CELLS = 90_000


def _cache_path(url: str, payload: dict | None) -> Path:
    key = url + ("" if payload is None else json.dumps(payload, sort_keys=True))
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    stem = url.rsplit("/", 1)[-1].replace(".px", "") or "root"
    return RAW_DIR / f"{stem}_{h}.json"


def _request(url: str, payload: dict | None = None, use_cache: bool = True) -> dict | list:
    cache = _cache_path(url, payload)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            if payload is None:
                resp = requests.get(url, timeout=60)
            else:
                resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    # PxWeb returns a UTF-8 BOM on data responses
    data = json.loads(resp.content.decode("utf-8-sig"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def list_dir(subpath: str = "") -> list[dict]:
    url = BASE_URL + (f"/{subpath}" if subpath else "")
    return _request(url)


def table_meta(table_path: str) -> dict:
    """Metadata (variables + value lists) for a table, e.g. '1_vnv/1_vnv/VIS01000.px'."""
    return _request(f"{BASE_URL}/{table_path}")


def _tidy_from_json(data: dict) -> pd.DataFrame:
    """Convert PxWeb 'json' format response to a tidy DataFrame."""
    cols = data["columns"]
    dim_cols = [c for c in cols if c["type"] != "c"]
    val_cols = [c for c in cols if c["type"] == "c"]
    rows = []
    for rec in data["data"]:
        rows.append(rec["key"] + rec["values"])
    # A single content column is standardized to "value"; multiple keep their codes.
    val_names = ["value"] if len(val_cols) == 1 else [c["code"] for c in val_cols]
    df = pd.DataFrame(rows, columns=[c["code"] for c in dim_cols] + val_names)
    for name in val_names:
        df[name] = pd.to_numeric(df[name].replace({".": None, "..": None, "-": None}), errors="coerce")
    return df


def fetch_table(table_path: str, time_var: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Fetch ALL cells of a table as a tidy DataFrame, chunking over the time
    variable when the request would exceed the API cell limit."""
    meta = table_meta(table_path)
    variables = meta["variables"]
    url = f"{BASE_URL}/{table_path}"

    if time_var is None:
        time_var = next(
            (v["code"] for v in variables if v.get("time") or v["code"] in ("Mánuður", "Tími", "Ár")),
            variables[0]["code"],
        )

    total_cells = 1
    for v in variables:
        total_cells *= len(v["values"])
    tvar = next(v for v in variables if v["code"] == time_var)
    other_cells = total_cells // len(tvar["values"])
    chunk_len = max(1, MAX_CELLS // other_cells)

    frames = []
    tvals = tvar["values"]
    for i in range(0, len(tvals), chunk_len):
        chunk = tvals[i : i + chunk_len]
        query = {
            "query": [
                {
                    "code": v["code"],
                    "selection": {
                        "filter": "item",
                        "values": chunk if v["code"] == time_var else v["values"],
                    },
                }
                for v in variables
            ],
            "response": {"format": "json"},
        }
        frames.append(_tidy_from_json(_request(url, query, use_cache=use_cache)))
    df = pd.concat(frames, ignore_index=True)

    # Attach human-readable labels for each dimension
    for v in variables:
        mapping = dict(zip(v["values"], v["valueTexts"]))
        if v["code"] in df.columns:
            df[v["code"] + "_text"] = df[v["code"]].map(mapping)
    return df


def discover_tables() -> dict:
    """Walk the VNV directory tree and return {table_id: {path, title, updated}}."""
    found = {}

    def walk(subpath: str):
        for entry in list_dir(subpath):
            child = f"{subpath}/{entry['id']}" if subpath else entry["id"]
            if entry["type"] == "l":
                walk(child)
            else:
                found[entry["id"].replace(".px", "")] = {
                    "path": child,
                    "title": entry["text"],
                    "updated": entry.get("updated"),
                }

    walk("1_vnv")
    return found
