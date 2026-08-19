# VNV forecast model (Icelandic CPI)

Bottom-up forecast of the vísitala neysluverðs (VNV): every component is
forecast separately and aggregated with Hagstofa's published, price-updated
weights through a reconstruction engine verified against every published month.

Built from `PLAN_1.md` (see `PLAN.md`). Phases 1–2 complete and gated; the
forecast layer is the v0 baseline of Phase 5.

## Read this first — the notebooks

| Notebook | What it shows |
|---|---|
| `notebooks/01_data_and_weights.ipynb` | All data straight from the PxWeb API; **the weights used to predict the CPI** (basket vogir vs monthly price-updated Vægi); anchor checks against press-release figures |
| `notebooks/02_reconstruction_check.ipynb` | **The hard gate**: the published headline is rebuilt from components for every month 2019–2026, chain-link months included, plus a negative control proving the check has teeth |
| `notebooks/03_forecast.ipynb` | 12-month forecast: per-component models with visible parameters, weight × m/m contribution tables, bootstrap fan chart, verðtrygging (t+2) path, honest walk-forward backtest |
| `notebooks/04_nowcast.ipynb` | Phase 3 observables: Gasvaktin fuel calibration (2016–, R²≈0.84) feeding CP0722; the January-2026 lesson (fuel observed correctly, km-charge missed → fiscal calendar); status of pending grocery/airfare feeds |
| `notebooks/05_imputed_rent.ipynb` | Phase 4: HMS new-contract rents → Hagstofa stock-based CP042 as a distributed lag; post-break EWMA + HMS tilt beats RW and AR(1) OOS; regime-break diagnostic; 12-month rent path |

Notebooks are committed **with outputs** so they read like a report. To re-run:

```
.venv\Scripts\python -m jupyter lab notebooks
```

## Dashboard

```
.venv\Scripts\python -m streamlit run streamlit_app.py
```

Five tabs: **Núspá** (current-month nowcast with observed-vs-model source per
component, live pump prices), **12 mánaða spá** (fan chart, contributions,
verðtrygging path), **Vogir** (the weights), **Endurbygging** (live gate
status), **Gagnalindir** (feed status + fuel calibration). Same `src/vnv` code
as the notebooks; Hagstofa data cached 1 hour.

Data is cached in `data/raw/` (git-ignored); delete the cache to force fresh
API fetches. `scripts/build_notebooks.py` regenerates and re-executes all
three notebooks from source.

## Layout

```
src/vnv/
  px_client.py     PxWeb API client: discovery, fetching, chunking, caching
  ingest.py        table -> tidy DataFrame loaders (VIS01000/01004/01300/01301/01306/01308)
  reconstruct.py   Laspeyres reconstruction + chain-link handling + rounding budgets
  models.py        v0 component models, aggregation, bootstrap fan chart
scripts/           runnable checks (each mirrors a notebook section) + notebook builder
config/px_tables.yaml   table IDs discovered from the API (log, not input)
```

## Key domain facts encoded (from PLAN_1.md)

- Published monthly **Vægi %** is the price-updated value share; contribution
  identity: m/m = Σ vægi(t−1) × change(t). At base changes (April ≤2024,
  January ≥2025) vægi(t−1) belongs to the old basket — de-updating vægi(t) by
  the component's own change handles all months with one formula.
- **Reiknuð húsaleiga** switched to rental equivalence June 2024: modeled on
  post-break sample only.
- COICOP2018 from January 2026; `VIS01308` is Hagstofa's own splice to 1997 —
  no home-made bridging.
- The API stores 1dp indices / 2dp weights & changes: all reconstruction
  tolerances are rounding-aware (the plan's <0.005pp is below the information
  content of the published data).
- VNV of month t settles verðtrygging in month t+2 (verified in notebook 03).

## Phase 3 status

- **Fuel** — live: Gasvaktin history (2016–) calibrated onto CP0722; the h=1
  nowcast reads today's pump prices (`vnv/fuel.py`, `vnv/nowcast.py`).
- **Groceries** — collecting: `kronan_price_history` change-log table + pg_cron
  (daily 08:30 UTC, days 1–15 only) applied in the home_app Supabase project
  2026-08-19, seeded with 4,413 SKUs. Export via
  `scripts/export_kronan_history.py` (needs SUPABASE_URL + service key), then
  `vnv/groceries.py` builds the matched-model food index. First usable m/m:
  September 2026.
- **Airfares** — framework only (`vnv/airfares.py`): quote-basket design and
  index math defined; needs a collection path (fares API key, browser scraper,
  or manual weekly quotes). The binding constraint on h=1 accuracy.
- **Fiscal calendar** — `config/fiscal_calendar.yaml`; Jan-2026 steps realized
  and quantified from published data.

## Phase 4 status — done

Imputed-rent model live (`vnv/rent.py`, `vnv/hms.py`): HMS leiguvísitala and
kaupvísitala auto-load from HMS's open CSVs (cached). CP042 m/m modeled post-June-
2024 only as EWMA persistence + a shrunk tilt on lag-1..3 HMS new-contract rents;
OOS RMSE 0.310 beats random walk (0.362) and AR(1) (0.339). Integrated via
`models.forecast_components(..., hms_rent_mm=, sub_spliced=)` so the headline
forecast and dashboard route CP042 through it. `hms.snapshot()` accrues input
vintages going forward.

## Not yet built (Phases 5–7)

Wage calendar + ARDL blocks (FX/ULC pass-through), BVAR + MinT reconciliation,
analyst / Seðlabanki / breakeven benchmark table.
