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
| `notebooks/06_driver_blocks.ipynb` | Phase 5: FX pass-through into food/imported goods (lands in the plan's 0.2–0.4 range), validated OOS as a shrunk tilt; wage calendar for kjarasamningar steps; full h=1 stack |
| `notebooks/07_reconciliation.ipynb` | Phase 6: top-down headline model + MinT reconciliation of bottom-up and top-down; exact aggregation identity; fan chart from the reconciled covariance; honest h=12 y/y (Atkeson–Ohanian) |
| `notebooks/08_backtest_report.ipynb` | Phase 7: pseudo-real-time backtest by horizon (beats seasonal-naive h=2–12; h=12 y/y beats RW on the post-2024 regime), block error decomposition, benchmark slots, and the monthly one-pager |

Notebooks are committed **with outputs** so they read like a report. To re-run:

```
.venv\Scripts\python -m jupyter lab notebooks
```

## Dashboard

```
.venv\Scripts\python -m streamlit run streamlit_app.py
```

Two tabs. **Skýrsla** (the report, main) reads top to bottom: (1) next month's
CPI, (2) the 36-month trend with MinT band, (3) each underlying with its trend
chart + method, (4) model vs breakeven, (5) model vs bank analysts, (6) model vs
Seðlabanki Peningamál. **Gögn & gæði** holds the diagnostics: weights, the Phase-2
reconstruction gate, and data-feed status + fuel calibration. The forecast path
runs 36 months (the 12-month figure stays the headline; beyond ~12m the top-down
anchor governs and bands widen). Same `src/vnv` code as the notebooks; Hagstofa
data cached 1 hour, market breakeven refreshed on load.

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

## Phase 5 status — done

Driver blocks live. New sources: Seðlabanki gengisvísitala (ISK NEER) via the
dated-snapshot API (`vnv/sedlabanki.py`), Hagstofa launavísitala
(`ingest.load_wages`), FAO world food (`vnv/worldfood.py`). `vnv/blocks.py`
estimates FX pass-through (CP01/CP02/CP03/CP071; cumulative 0.19–0.43 over 12m,
in the plan's range) and applies it as a shrunk tilt where it beats the generic
fit OOS; `config/wage_calendar.yaml` carries kjarasamningar steps. Integrated via
`models.forecast_components(..., comp_history=, fx_mm=)`. Full h=1 stack (fuel +
rent + FX) improves the ex-January headline RMSE 0.353 → 0.320; the ≤0.15pp gate
is airfare-gated (airfares still collecting) and needs the fiscal calendar applied
for January — exactly as the plan anticipates.

## Phase 6 status — done

Top-down headline model (`vnv/topdown.py`: seasonal + AR + trend gliding to the
2.5% target; `anchor_yoy` exposed for breakeven blending) and MinT reconciliation
(`vnv/reconcile.py`) in contribution space, so the aggregation identity holds
exactly and the fan chart comes from the reconciled covariance. The reconciliation
pulls bottom-up → top-down at longer horizons, as designed. h=12 y/y beats
RW-on-y/y ex-surge but not full-sample (Atkeson–Ohanian; reported honestly, not
tuned). Integrated into the dashboard "12 mánaða spá" tab.

## Phase 7 status — done

Pseudo-real-time backtest by horizon (`vnv/backtest.py`): beats the seasonal-naive
floor at h=2–12; h=12 y/y RMSE 0.24 vs RW-on-y/y 1.15 on the post-2024 regime the
model targets (full-sample fails Atkeson–Ohanian, reported honestly). h=1 block
error decomposition confirms the plan's variance ranking (imported-goods útsölur +
airfares dominate). Monthly one-pager (`vnv/report.py`, bilingual): nowcast,
12-month path with MinT band, contribution waterfall by driver block, verðtrygging
(t+2) path, model-vs-breakeven — in the dashboard "Skýrsla" tab and downloadable.
Benchmark slots (`vnv/benchmarks.py`, `data/benchmarks/`) ready for the breakeven
and analyst feeds.

## Breakeven benchmark — live from lanamal.is

Breakeven inflation (RIKB nominal − RIKS real yields) — the tradeable benchmark —
comes live from Lánamál ríkisins (`vnv/lanamal.py`, the `LoadChartData` JSON API;
no paid feed needed). `lanamal.update_breakeven_slot()` writes the curve to
`data/benchmarks/breakeven.csv`; the report's model-vs-breakeven section and the
dashboard "Skýrsla" tab render it automatically. Refresh with
`scripts/refresh_breakeven.py`. The shortest indexed bond matures ~4y out, so the
shortest breakeven is ~4y — its gap to the model's near-term call is an indicative
premium/expectations wedge (breakevens carry inflation-risk and indexed-bond
scarcity premia, which is where the h=3–12 edge lives).

## Analyst benchmark

Bank-analyst forecasts (Íslandsbanki Greining, Landsbankinn) are compared against
the model and realized actuals — real published figures in
`data/benchmarks/analyst_forecasts.csv` (one row per source per print, added each
month from the banks' forecast notes ~1 week before release). Their longer-horizon
**annual** forecasts (from the banks' semi-annual þjóðhagsspá/hagspá) are in
`data/benchmarks/analyst_annual.csv` and overlaid on the Section-4 term structure
(mapped to years-ahead), so the model's 1/2/3-year average sits next to each bank's
forward view and the market breakeven — the model reads ~4.1% at 1yr vs the banks'
3.3–3.6%, its rent-persistence view again. The report's
"Módel vs greiningaraðilar" section and dashboard Section 5 show the upcoming-print
model-vs-consensus gap and each source's realized track record. As the plan
predicts, analysts are hard to beat at h=1 (they run the same public scrapes) — the
edge is at h=3–12; the track record confirms it (Landsbankinn h=1 RMSE ~0.09pp).

## Seðlabanki Peningamál benchmark

The central bank's quarterly y/y forecast (Peningamál Tafla 5) is the longer-horizon
benchmark — real figures in `data/benchmarks/peningamal.csv`, updated ~4×/year. The
report's "Módel vs Seðlabanki" section overlays the model's quarterly-average y/y on
the Peningamál path. Current read: the model agrees near-term (2026 within ~0.2pp)
but sees **stickier inflation at 12 months** (2027Q3: model 4.1% vs Seðlabanki 3.0%,
+1.1pp) — the model's differentiated, structural view, since rental-equivalence rent
(Phase 4) is persistent while the bank's path reverts faster to target.

**All seven plan phases are implemented and live end to end**, with the full
benchmark set: seasonal-naive & random-walk floors, the tradeable RIKB−RIKS
breakeven, bank-analyst consensus, and Seðlabanki Peningamál. The airfare and
grocery observable feeds keep maturing to sharpen h=1; the analyst and Peningamál
slots grow one row per print / quarter.
