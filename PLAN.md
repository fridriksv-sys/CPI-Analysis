# VNV Forecast Model — Implementation Plan

Bottom-up predictive model for the Icelandic CPI (vísitala neysluverðs, "VNV"),
producing a 1-month nowcast and a 12-month path by forecasting each component
and re-aggregating with published weights.

**Primary use case:** the fund holds indexed instruments (RIKS, sjóðfélagalán,
verðtryggð skuldabréf). VNV published in month *t* applies to indexation in
month *t+2*, so months 1–2 are already known. **The economic value of this model
is at h=3 to h=12**, compared against breakeven inflation (RIKS vs RIKB).

---

## 0. Rules of engagement for the agent

- **Do not skip Phase 2.** It is a hard gate. No forecasting code is written
  until historical reconstruction ties out.
- **Never hardcode a PxWeb table ID or a weight you have not fetched.** Discover
  tables via the API and assert on the response. If a source is unreachable,
  STOP and report — do not substitute plausible numbers.
- **No synthetic/placeholder data anywhere**, including in tests. Tests use real
  fetched fixtures committed to `data/fixtures/`.
- After each phase, STOP, print the acceptance-criteria results, and wait for
  review before proceeding.
- Icelandic column names and identifiers are fine and preferred where they match
  the source (`reiknud_husaleiga`, `flugfargjold`, etc.). Code comments in English.
- Commit at the end of each phase with the phase number in the message.

---

## 1. Domain constraints the agent must encode

These are non-obvious and will silently corrupt results if ignored.

### 1.1 Structural break: imputed rent, June 2024
Hagstofa replaced the simple user-cost method (einfaldur notendakostnaður:
house price index + real mortgage rates) with **rental equivalence
(leiguígildi)** from June 2024. The source is contracts assessed as market rent
in HMS's rental database plus older registered contracts, covering the whole
country and **all contracts in force**, not just new ones.

Consequences:
- `042 Reiknuð húsaleiga` is ~21–22% of the basket — the single largest component.
- Pre-June-2024 history for this series comes from a different data-generating
  process. Do **not** estimate on a sample spanning the break without a regime
  dummy or, preferably, restrict to post-2024 with shrinkage priors.
- HMS's leiguvísitala is capital-area-only and covers **only newly-signed
  contracts**. Hagstofa's measure is nationwide and stock-based. Model the
  mapping as an explicit distributed lag from new-contract rents into the rent
  stock. They are not substitutes.

### 1.2 Classification break: COICOP2018, January 2026
- January 2026 moved to COICOP2018 (13 divisions) on a new **December 2025 base**,
  built from the 2022–2024 household expenditure survey (rannsókn á útgjöldum
  heimilanna), three-year averaged.
- Hagstofa publishes a **bridging table** linking COICOP2018 subindices to the
  older classification. Use it. Do not construct your own mapping.
- Splicing formula Hagstofa documents: old-base value × (new-base value / 100).
  Request extra decimal places from PxWeb ("Breyta og reikna") — 1 dp will not
  clear the Phase 2 gate.

### 1.3 Rebasing moved to January
Historically the basket was renewed in **March** (since 1997). It moved to
**January** as of the January 2025 index. Any chain-link logic must handle both
conventions depending on the year. Basket renewal itself does not move the index
month-on-month.

### 1.4 Fuel tax overhaul, January 2026
Olíugjald and fuel excise (vörugjöld á eldsneyti) were **abolished**, carbon tax
(kolefnisgjald) raised, and a **kilometre charge (kílómetragjald)** introduced
for all vehicles. Vehicle excise changed and the EV subsidy was cut.
- Pass-through equations fitted on pre-2026 data have the wrong tax wedge.
- The km charge is a usage levy, not a pump-price component — check which CPI
  subindex it lands in before assuming.

### 1.5 The index is never revised retroactively
A published VNV value stands permanently; corrections are absorbed into the next
release. (Example: a wrong fuel price from one retailer in January 2026 was
handled in the February release, not by restating January.)
- Good: no vintage problem on the **target**.
- But there **is** a vintage problem on **inputs** (HMS indices publish with a
  lag). The backtest must be vintage-aware on the input side.

### 1.6 Collection and publication timing
- Prices are collected over roughly the **first half of the month**.
- The release lands ~23rd–29th of the same month.
- The nowcast is therefore produced around day 15–18, when a large share of the
  basket is *observable* rather than forecastable.

---

## 2. Repo layout

```
vnv-model/
  pyproject.toml
  CLAUDE.md                     # short version of §0 + §1 for agent context
  PLAN.md                       # this file
  config/
    blocks.yaml                 # driver-based component grouping (§3)
    fiscal_calendar.yaml        # dated administered price steps
    wage_calendar.yaml          # kjarasamningar step schedule
  src/vnv/
    ingest/
      hagstofa_px.py            # PxWeb API client
      hagstofa_releases.py      # press-release scraper -> implied weights
      hms.py                    # íbúðaverðsvísitala, leiguvísitala
      sedlabanki.py             # FX, policy rate, breakevens
      fuel.py                   # daily pump prices
      airfares.py               # fare quotes for the collection window
      groceries.py              # Krónan / Bónus / Nettó
    index/
      reconstruct.py            # rebuild published VNV from subindices
      chainlink.py              # Jan (and pre-2025 Mar) base changes
      weights.py                # published + reverse-engineered weights
    models/
      blocks/                   # one module per driver block
      topdown.py                # BVAR / UC on headline
      reconcile.py              # MinT hierarchical reconciliation
    backtest/
      vintages.py
      evaluate.py
    report/
      monthly.py                # one-pager output
  tests/
  data/
    raw/ interim/ fixtures/
  notebooks/
```

Stack: Python 3.11+, `pandas`, `requests`, `statsmodels`, `linearmodels`,
`pymc` or `bambi` (short post-2024 samples need shrinkage),
`hierarchicalforecast` (Nixtla) for MinT, `pytest`, `ruff`.

---

## 3. Component blocks (config/blocks.yaml)

Group by **driver**, not by COICOP. Indicative weights — the agent must fetch
actual values, these are only for sanity-checking magnitudes.

| Block | ~Weight | Driver | h=1 predictability |
|---|---|---|---|
| `reiknud_husaleiga` | ~21.5% | Market rents (HMS stock) | High |
| `greidd_husaleiga` | ~3% | Same, contract-lagged | High |
| `hiti_rafmagn_veitur` | ~4% | Administered tariffs | Near-certain |
| `eldsneyti` | ~2.5% | Brent/crack × ISKUSD + tax | Near-certain (observable) |
| `innfluttar_vorur` | ~20% | ISK NEER pass-through | Medium |
| `matur_drykkur` | ~15% | FX, world food, domestic ag | Medium |
| `innlend_thjonusta` | ~15% | Wages / ULC, inertia | Medium |
| `ferdalog` (flug, gisting, pakkaferðir) | ~4% | Seasonal, capacity, jet fuel | Low, high variance |
| `opinber_thjonusta` | ~5% | Fjárlög, municipal budgets | Near-certain, timing-driven |
| `afengi_tobak_annad` | ~10% | Excise + FX | Medium |

Each block maps to an explicit list of COICOP2018 subindex codes. Weights must
sum to 1.0 and the mapping must be exhaustive and non-overlapping — assert both.

This grouping deliberately mirrors Seðlabanki's innlendar / innfluttar / húsnæði
/ þjónusta decomposition in Peningamál so forecasts are directly comparable.

---

## Phase 1 — Ingestion and the weight panel

**Tasks**
1. PxWeb client against `https://px.hagstofa.is/pxis/api/v1/is/Efnahagur/visitolur/`.
   Enumerate the directory tree via the API; discover the VNV subindex tables,
   the weights (vogir) table, and the COICOP2018 ↔ old-classification bridging
   table. Log the discovered IDs to `config/px_tables.yaml`. Cache all responses
   to `data/raw/`.
2. Pull full subindex history at maximum available decimal precision.
3. Splice pre-2026 (old COICOP) to post-2026 (COICOP2018) using the published
   bridging table.
4. Scrape the press-release archive. URLs follow the pattern
   `https://hagstofa.is/utgafur/frettasafn/verdlag/visitala-neysluverds-i-{manudur}-{ar}/`.
   Parse the *áhrif á vísitöluna* figures.
5. **Reverse-engineer effective weights**: for each mentioned component,
   `implied_weight = áhrif_pp / pct_change`. Build a monthly panel. This gives
   price-updated weights that the published annual vogir do not.

   Sanity anchors from recent releases (the agent should reproduce these):
   `föt og skór ≈ 3.6%`, `húsbúnaður ≈ 4.3%`, `matur ≈ 15.0%`,
   `flugfargjöld ≈ 2.5–2.6%`, `reiknuð húsaleiga ≈ 21.7%`,
   `bensín ≈ 1.7%`, `dísel ≈ 0.8%`, `gistiþjónusta ≈ 0.6%`.

**Acceptance criteria**
- Subindex history loads from 2015 to current with no gaps.
- Spliced series are continuous across Jan 2026 (no artificial jump at the break).
- Implied-weight panel reproduces the anchor figures above within ±0.2pp.
- All network calls cached; the test suite runs offline from fixtures.

---

## Phase 2 — Index reconstruction (HARD GATE)

Rebuild the published headline VNV from subindices and weights. VNV is a
fixed-base Laspeyres index, chain-linked at each base change, with weights
price-updated within the year. Naively computing `Σ(w_i × Δp_i)` with static
published weights drifts by tens of basis points over a year — this is the most
common way to get this project wrong.

**Tasks**
1. Implement `reconstruct.py` with correct within-year weight price-updating.
2. Implement `chainlink.py` handling January base changes (2025 onwards) and
   March base changes (pre-2025).
3. Cross-check against the implied-weight panel from Phase 1.
4. Also reconstruct **VNV án húsnæðis** (excludes 041, 042, 043, 044 but
   **retains 045 hiti og rafmagn**) as an independent check on the mapping.

**Acceptance criteria — all must pass**
- Reconstructed headline VNV matches published values to **< 0.02 index points**
  for every month from January 2019 to the latest print.
- Reconstructed m/m % matches published m/m % to **< 0.005pp**.
- VNV án húsnæðis reconstruction passes the same tolerances.
- Chain-link months (every January since 2025, every March 2019–2024) pass with
  no special-casing beyond the documented base-change logic.

**If these do not pass, STOP.** Do not proceed to forecasting. A model that
cannot rebuild the past is not forecasting the instrument that settles the bonds.

---

## Phase 3 — Observable-input nowcast layer

Roughly 30% of the basket can be measured rather than modelled by day 15.

**Priority order (by variance contribution, highest first):**

1. **Airfares (`flugfargjöld til útlanda`).** Highest leverage in the whole
   model. ~2.5% weight but ±10–20% monthly moves: June 2026 was a +0.94% print
   of which airfares alone contributed +0.52pp on a +20.1% move; January 2026
   contributed −0.27pp on −10.8%. Scrape Icelandair and PLAY fare quotes for
   routes and booking windows matching Hagstofa's collection period. Calibrate
   the scraped index against the published subindex.
2. **Fuel.** Scrape daily pump prices (bensinverd.is, GSMbensín). Match
   Hagstofa's outlet set and average over the collection window. Post-Jan-2026
   tax structure only.
3. **Administered prices.** Build `fiscal_calendar.yaml` from fjárlög, municipal
   fee schedules, and utility tariffs (Orkuveitan, HS Veitur). Dated step
   changes with the affected subindex and expected pp impact. January and July
   are the dense months.
4. **Groceries.** Scrape Krónan / Bónus / Nettó. Weight to Hagstofa's food
   basket structure. (Note: prior Krónan price-scraping work exists and should
   be reused rather than rebuilt.)
5. **Seasonal residual** — clothing, furniture, útsölur cycle. Estimated seasonal
   factors plus an AR term. Jan/Feb sales dynamics are the second-largest
   variance source after airfares.

**Acceptance criteria**
- Each scraper has a backfilled history long enough to calibrate against the
  corresponding published subindex (minimum 24 months where obtainable).
- Calibration regression of scraped index on published subindex reports R² and
  residual SD per component; these feed the error budget.

---

## Phase 4 — Imputed rent model

The largest component and the one with the shortest usable sample.

**Tasks**
1. Ingest HMS leiguvísitala and íbúðaverðsvísitala, vintage-tagged by actual
   publication date.
2. Model the mapping from HMS new-contract rents (capital area) to Hagstofa's
   nationwide stock-based measure as a distributed lag. Expect substantial
   smoothing — the stock turns over slowly.
3. Estimate on **post-June-2024 data only**, with informative priors, or on the
   full sample with an explicit regime break. Justify the choice in a docstring
   and report both.
4. Note the regime implication for the 12-month path: rents are far more
   persistent than house prices were under user cost, so the largest component
   is now *more* forecastable and headline inflation is *less* policy-rate
   sensitive than pre-2024 history implies. Do not let a long-sample model
   inherit the old rate sensitivity.

**Acceptance criteria**
- Out-of-sample RMSE on `reiknuð húsaleiga` m/m beats a random walk and an AR(1)
  over the post-2024 period.
- Backtest respects HMS publication lags (no look-ahead).

---

## Phase 5 — Remaining blocks and full h=1

**Tasks**
1. ARDL per remaining block: imported goods on ISK NEER (expect partial
   pass-through, roughly 0.2–0.4 over 12 months — estimate, don't assume), food
   on FX + world food prices, domestic services on wages/ULC with inertia.
2. Encode `wage_calendar.yaml`: kjarasamningar increases as dated step shifts to
   the domestic services block. Do **not** let a regression discover these from
   the data.
3. Aggregate through the Phase 2 reconstruction engine.

**Acceptance criteria**
- Full h=1 headline nowcast RMSE on m/m ≤ **0.15pp** out of sample.
- ≤ 0.10pp is not a realistic target given airfare and sales noise; if the
  backtest reports better than 0.08pp, treat it as evidence of look-ahead
  leakage and investigate before celebrating.

---

## Phase 6 — 12-month path and reconciliation

Bottom-up alone will fail at longer horizons: component errors compound and the
drivers themselves become unknown.

**Tasks**
1. **Bottom-up path**: ARDL blocks driven by forward-looking inputs — Brent
   forward curve, ISK forwards, dated fiscal steps, wage agreement schedule,
   seasonal factors.
2. **Top-down model**: small BVAR or unobserved-components model on headline,
   with the long end anchored between the 2.5% target and breakeven-implied
   inflation.
3. **Reconcile with MinT** (Wickramasuriya / Athanasopoulos / Hyndman). The
   aggregation weights are known, which is precisely the setting MinT is built
   for. Expect it to pull the bottom-up path toward the top-down at longer
   horizons — that is the correct behaviour, not a bug.
4. Produce fan charts from the reconciled covariance, not from naive
   component-error addition.

**Acceptance criteria**
- Reconciled path is arithmetically consistent with the component paths through
  the Phase 2 engine (aggregation identity holds exactly).
- h=12 y/y RMSE beats random-walk-on-y/y over the backtest sample.

---

## Phase 7 — Backtest, benchmarks, and output

**Backtest design**
- Pseudo-real-time, expanding window, vintage-aware on inputs.
- Sample from 2015 where possible; imputed rent block restricted per Phase 4.
- Report RMSE and MAE by horizon (h=1…12) and a contribution-level error
  decomposition so it is clear *which block* is driving misses.

**Benchmarks — all of these**
| Benchmark | Purpose |
|---|---|
| Seasonal naive on m/m | Floor |
| Random walk on y/y | Floor |
| Bank analysts (Greining Íslandsbanka, Landsbankinn Hagfræðideild, Arion) | Published ~1 week before the print |
| Seðlabanki Peningamál | Quarterly official forecast |
| Breakeven inflation (RIKS vs RIKB) | The tradeable benchmark |

**Be honest in the write-up:** the model will probably *not* beat analyst
consensus at h=1 — they run the same scrapes. Do not tune until it appears to.
The edge is at h=3–12, where breakevens are contaminated by an inflation risk
premium and an indexed-bond scarcity premium.

**Monthly output** (`report/monthly.py`): point forecast and fan chart, m/m
contribution waterfall by block, implied verðtrygging path (index in month *t*
→ indexation in *t+2*), and model-vs-breakeven decomposed into expectations and
premium. English and Icelandic labels.

---

## Anti-patterns — do not do these

- Aggregating with static published weights and no price-updating.
- Estimating imputed rent across the June 2024 break without a regime split.
- Treating HMS leiguvísitala as a drop-in for Hagstofa's reiknuð húsaleiga.
- Fitting fuel pass-through on a sample spanning January 2026.
- Building forecasts before Phase 2 passes.
- Using low-precision (1 dp) PxWeb output.
- Silently filling a failed data fetch with an estimate.
- Tuning until h=1 beats consensus — that is overfitting to a benchmark you
  cannot beat structurally.
