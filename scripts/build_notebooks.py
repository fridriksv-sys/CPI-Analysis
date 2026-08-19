"""Generate and execute the project notebooks.

Each notebook is built from markdown + code cells here, then executed with
nbclient so the committed .ipynb files carry their outputs.
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

BOOTSTRAP = '''\
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 250)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}" if abs(x) < 1e6 else f"{x:,.0f}")

# Okabe-Ito palette (colorblind-safe); recessive grid per house chart style
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "pink": "#CC79A7", "sky": "#56B4E9", "black": "#222222"}
plt.rcParams.update({
    "figure.figsize": (11, 4.5), "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
'''


def build(path: Path, cells: list[tuple[str, str]]):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    client = NotebookClient(nb, timeout=1200, resources={"metadata": {"path": str(NB_DIR)}})
    client.execute()
    nbf.write(nb, path)
    print("built + executed:", path.name)


# ============================================================================
# Notebook 1 — data and weights
# ============================================================================
nb1 = [
("md", """\
# 01 — Gögn og vogir (data and the CPI weights)

**Purpose.** Load every input of the VNV model directly from Hagstofa's PxWeb API,
show what each table contains, and lay out the **weights used to aggregate component
forecasts into the headline CPI** (vísitala neysluverðs, VNV).

**How to error-check this notebook**
- Every series comes from the API, cached in `data/raw/` (delete the cache to force a re-fetch).
- No table ID is trusted blindly: the first cell *discovers* the tables from the API directory tree.
- The anchor table at the bottom compares the fetched weights against figures quoted in
  Hagstofa press releases (from `PLAN_1.md`) — every row should agree.

| Table | Content | Coverage |
|---|---|---|
| `VIS01000` | Headline VNV + VNV án húsnæðis, index levels and published changes | 1988M05– |
| `VIS01300` | COICOP2018 subindices: index, **monthly weight (Vægi %)**, m/m change, effect (áhrif) | 2025M01– |
| `VIS01301` | Same measures under the old classification (IS codes) | 2002M08–2025M12 |
| `VIS01306` | Basket weights (vogir), December 2024 & December 2025 bases, **per 10 000** | annual |
| `VIS01308` | Subindex history spliced to COICOP2018 codes **by Hagstofa** (the bridging table) | 1997M03– |
| `VIS01004` | Vísitala neysluverðs til verðtryggingar (settles indexed instruments at t+2) | 1979M06– |
"""),
("code", BOOTSTRAP),
("code", '''\
from vnv import px_client, ingest

# Discover every VNV table from the API tree - nothing hardcoded
tables = px_client.discover_tables()
pd.DataFrame(tables).T
'''),
("code", '''\
head = ingest.load_headline()          # VIS01000
vt = ingest.load_verdtrygging()        # VIS01004
sub_new = ingest.load_sub_new()        # VIS01300
panel_old = ingest.load_panel_old()    # VIS01301
weights_new = ingest.load_weights_new()# VIS01306
spliced = ingest.load_sub_spliced()    # VIS01308

print("headline:", head.index.min(), "-", head.index.max())
print("new subindex panel:", sub_new.manudur.min(), "-", sub_new.manudur.max(),
      f"({sub_new.code.nunique()} codes)")
print("old panel:", panel_old.manudur.min(), "-", panel_old.manudur.max(),
      f"({panel_old.code.nunique()} codes)")
print("spliced history:", spliced.manudur.min(), "-", spliced.manudur.max(),
      f"({spliced.code.nunique()} codes)")
'''),
("code", '''\
yy = head[("CPI", "change_A")]["2015":]
yy_lh = head[("CPILH", "change_A")]["2015":]
fig, ax = plt.subplots()
ax.plot(yy.index.to_timestamp(), yy, color=C["blue"], lw=2, label="VNV y/y")
ax.plot(yy_lh.index.to_timestamp(), yy_lh, color=C["orange"], lw=2, label="VNV án húsnæðis y/y")
ax.axhline(2.5, color=C["black"], lw=1, ls="--", alpha=0.6)
ax.annotate("2.5% markmið", xy=(yy.index.to_timestamp()[6], 2.7), fontsize=9, alpha=0.8)
ax.set_title("Verðbólga (y/y, %)")
ax.legend(frameon=False)
plt.show()
'''),
("md", """\
## The weights

Two kinds of weight exist and the difference matters:

1. **Basket weights (vogir, `VIS01306`)** — the December basket from the household
   expenditure survey, fixed quantities for the year, published **per 10 000**.
2. **Monthly price-updated weights (Vægi %, `VIS01300`/`VIS01301`)** — the same basket's
   *value shares at current prices*. These evolve every month as relative prices move and
   are the weights that actually enter each month's Laspeyres contribution identity:

$$\\text{headline m/m} = \\sum_i w_{i,t-1}\\,g_{i,t}, \\qquad w_{i,t-1} = \\text{Vægi at } t-1$$

Notebook 02 verifies this identity against every published month. The forecast in
notebook 03 starts from the latest published Vægi and price-updates it along the path.
"""),
("code", '''\
DIV = [f"CP{i:02d}" for i in range(1, 14)]
last_m = sub_new.manudur.max()
latest = sub_new[sub_new.manudur == last_m].set_index("code")

w_base = weights_new.set_index("code")["2025M12"] / 100  # per-10000 -> %
div_tbl = pd.DataFrame({
    "heiti": latest.loc[DIV, "heiti"],
    "vog des-2025 (%)": w_base[DIV],
    f"vægi {last_m} (%)": latest.loc[DIV, "vaegi"],
})
div_tbl["breyting (pp)"] = div_tbl.iloc[:, 2] - div_tbl.iloc[:, 1]
print(f"sums: basket {div_tbl.iloc[:, 1].sum():.2f} | current {div_tbl.iloc[:, 2].sum():.2f}")
div_tbl.round(2)
'''),
("code", '''\
d = div_tbl.sort_values(f"vægi {last_m} (%)")
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.barh(d.heiti.str.slice(0, 48), d[f"vægi {last_m} (%)"], color=C["blue"], height=0.62)
for i, v in enumerate(d[f"vægi {last_m} (%)"]):
    ax.text(v + 0.25, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_title(f"Vægi í vísitölu neysluverðs, {last_m} (%)")
ax.grid(axis="y", alpha=0)
plt.tight_layout(); plt.show()
'''),
("md", """\
## Forecast components

The model works at division level with two divisions split into groups: housing (CP04),
so that **reiknuð húsaleiga (CP042)** — the largest single component, with the June-2024
methodology break — is modeled on its own regime; and transport (CP07), so that
**eldsneyti (CP0722)** can take the observable fuel nowcast (notebook 04).
"""),
("code", '''\
from vnv import models

comp = latest.loc[models.COMPONENTS, ["heiti", "vaegi"]].rename(columns={"vaegi": f"vægi {last_m} (%)"})
assert abs(comp.iloc[:, 1].sum() - 100) < 0.05, "component weights must partition the basket"
print(f"component weights sum: {comp.iloc[:, 1].sum().round(2)} "
      f"({len(models.COMPONENTS)} non-overlapping components)")
comp.round(2)
'''),
("md", """\
## Anchor check — do the fetched weights reproduce the press-release figures?

`PLAN_1.md` quotes approximate implied weights from Hagstofa releases. The published
Vægi is price-updated and moves month to month (airfares' weight swings from 1.4% in
winter to 3.5% in July — that alone is a good sanity check that these are value shares,
not fixed weights). The test: each anchor must sit inside — or within ~1pp of — the range
its component's published weight actually took over 2025–2026.
"""),
("code", '''\
anchor_rows = [
    ("matur og drykkur (CP01)", "CP01", 15.0),
    ("föt og skór (CP03)", "CP03", 3.6),
    ("húsbúnaður (CP05)", "CP05", 4.3),
    ("reiknuð húsaleiga (CP042)", "CP042", 21.7),
    ("flugfargjöld til útlanda (CP07332)", "CP07332", 2.55),
    ("bensín (CP07222)", "CP07222", 1.7),
    ("dísel (CP07221)", "CP07221", 0.8),
    ("gistiþjónusta (CP112)", "CP112", 0.6),
]
rows = []
for name, code, plan_val in anchor_rows:
    s = sub_new[sub_new.code == code].set_index("manudur").vaegi
    dist = max(0.0, s.min() - plan_val, plan_val - s.max())
    rows.append({"component": name, "plan anchor (%)": plan_val,
                 "vægi range 2025–26 (%)": f"{s.min():.2f} – {s.max():.2f}",
                 f"vægi {last_m} (%)": s[last_m], "distance to range (pp)": dist,
                 "status": "í bili" if dist == 0 else "nálægt"})
anchor_tbl = pd.DataFrame(rows).set_index("component")
assert anchor_tbl["distance to range (pp)"].max() < 1.0, "anchor severely off - wrong component mapping?"
anchor_tbl.round(2)
'''),
("md", """\
Six of eight anchors fall inside the observed weight range. The two marked *nálægt*
are explained by basket vintage: the plan quotes ~15.0% for food and ~21.7% for imputed
rent from earlier releases/approximations (`PLAN_1.md` §1.1 itself says "~21–22%");
the COICOP2018 December-2025 basket puts them at 15.44% and 20.88%. The component
identification is unambiguous in every case.

## The June-2024 break in reiknuð húsaleiga

Weight history of imputed rent: old classification (IS042) until 2024, COICOP2018 (CP042)
from 2025. The June-2024 switch from user cost to rental equivalence *changed the
data-generating process*, which is why notebook 03 fits this component on post-break data only.
"""),
("code", '''\
w_old = panel_old[panel_old.code == "IS042"].set_index("manudur").vaegi["2015":]
w_new_s = sub_new[sub_new.code == "CP042"].set_index("manudur").vaegi
fig, ax = plt.subplots()
ax.plot(w_old.index.to_timestamp(), w_old, color=C["blue"], lw=2, label="IS042 (eldra kerfi)")
ax.plot(w_new_s.index.to_timestamp(), w_new_s, color=C["orange"], lw=2, label="CP042 (COICOP2018)")
brk = pd.Period("2024-06", "M").to_timestamp()
ax.axvline(brk, color=C["red"], lw=1.2, ls="--")
ax.annotate("leiguígildi tekur við\\njúní 2024", xy=(brk, w_old.max()), fontsize=9,
            color=C["red"], ha="right", xytext=(-8, -18), textcoords="offset points")
ax.set_title("Vægi reiknaðrar húsaleigu (%)")
ax.legend(frameon=False, loc="lower left")
plt.show()
'''),
]

# ============================================================================
# Notebook 2 — reconstruction gate
# ============================================================================
nb2 = [
("md", """\
# 02 — Endurbygging vísitölunnar (reconstruction check — the hard gate)

**Purpose.** Before any forecasting, prove that we can rebuild the *published* headline
VNV from its components and weights. A model that cannot rebuild the past is not
forecasting the instrument that settles the bonds (`PLAN_1.md`, Phase 2).

**The identity.** VNV is a fixed-base Laspeyres index. Component $i$'s contribution in
month $t$ is its basket-value share at $t-1$ prices times its m/m change. We recover that
share by *de-updating* the published month-$t$ share by the component's own change:

$$w^{pre}_{i,t} = \\frac{\\text{Vægi}_{i,t}/(1+g_{i,t}/100)}{\\sum_j \\text{Vægi}_{j,t}/(1+g_{j,t}/100)}\\times 100,
\\qquad \\widehat{\\text{m/m}}_t = \\sum_i w^{pre}_{i,t}\\, g_{i,t}/100$$

For ordinary months this equals Vægi at $t-1$ identically. In **base-change months**
(April up to 2024, January from 2025) Vægi$_{t-1}$ belongs to the *old* basket while the
index is computed on the new one — de-updating handles both cases with one formula,
no special-casing.

**Precision ceiling (deviation from the plan, documented).** The API stores indices at
1 decimal and weights/changes/effects at 2 decimals; there is no higher-precision
endpoint (verified against both `json` and `json-stat2` formats). The plan's tolerances
(<0.005pp m/m, <0.02 index points) therefore exceed the information content of the
published data. The achievable — and sufficient — gate is: **errors bounded by the
propagated rounding budget, mean error ≈ 0, and no cumulative drift beyond rounding.**
"""),
("code", BOOTSTRAP),
("code", '''\
from vnv import ingest, reconstruct

head = ingest.load_headline()
panel_old = ingest.load_panel_old()
sub_new = ingest.load_sub_new()
weights_new = ingest.load_weights_new()
'''),
("md", "## Gate 1 — old classification, 2019–2025 (84 months)"),
("code", '''\
r1 = reconstruct.mm_from_panel(panel_old, reconstruct.DIV_OLD)
r1 = r1[r1.index >= "2019-01"]
budget = reconstruct.rounding_budget_mm(pd.Series(100 / 12, index=reconstruct.DIV_OLD), 12)

print(f"months checked: {len(r1)}")
print(f"max |error|:  {r1.error.abs().max():.4f} pp   (worst-case rounding budget: {budget:.4f} pp)")
print(f"mean error:   {r1.error.mean():+.5f} pp  (no systematic bias)")
gate1 = r1.error.abs().max() <= budget
print("GATE 1:", "PASS" if gate1 else "FAIL")
r1.error.abs().groupby(r1.index.year).max().round(4).to_frame("max |error| pp")
'''),
("code", '''\
fig, ax = plt.subplots()
ax.plot(r1.index.to_timestamp(), r1.error * 100, color=C["blue"], lw=1.5)
ax.axhspan(-budget * 100, budget * 100, color=C["sky"], alpha=0.25, lw=0)
ax.annotate("rounding budget", xy=(r1.index.to_timestamp()[2], budget * 100 * 0.75),
            fontsize=9, alpha=0.8)
ax.set_title("Reconstruction error, m/m (basis points) — every month 2019–2025")
ax.set_ylabel("bp")
plt.show()
'''),
("md", """\
## Gate 2 — chain-link months get no special treatment

The worst historic failure mode: in April (old regime) and January (from 2025) the basket
switches. Show those months pass under the *same* formula.
"""),
("code", '''\
chain_months = [m for m in r1.index if (m.month == 4 and m.year <= 2024) or (m.month == 1 and m.year >= 2025)]
chain = r1.loc[chain_months, ["recon_mm", "published_mm", "error"]]
gate2 = chain.error.abs().max() <= budget
print("GATE 2 (chain months, same formula):", "PASS" if gate2 else "FAIL")
chain.round(4)
'''),
("md", """\
### Why this matters — the static-weight anti-pattern

Aggregating with the December basket weights *unadjusted all year* (the most common way
to get this wrong) drifts visibly. Same data, wrong weight handling:
"""),
("code", '''\
div = panel_old[panel_old.code.isin(reconstruct.DIV_OLD)]
year = div[(div.manudur >= "2023-01") & (div.manudur <= "2023-12")]
w_static = year[year.manudur == "2023-01"].set_index("code").vaegi  # frozen January weights

static_mm = year.pivot_table(index="manudur", columns="code", values="manadarbreyting")
naive = (static_mm[w_static.index] * w_static.values).sum(axis=1) / w_static.sum()
correct = r1.loc[naive.index, "recon_mm"]
pub = r1.loc[naive.index, "published_mm"]
cmp = pd.DataFrame({"static weights": naive, "price-updated": correct, "published": pub})
cmp["static error (pp)"] = cmp["static weights"] - cmp["published"]
print(f"static-weight cumulative 2023 error: {cmp['static error (pp)'].sum():+.3f} pp "
      f"(vs {(correct - pub).sum():+.4f} pp price-updated)")
cmp.round(3)
'''),
("md", "## Gate 3 — COICOP2018 era (2026–): fixed-base level aggregation"),
("code", '''\
w_dec25 = weights_new[weights_new.code.isin(reconstruct.DIV_NEW)].set_index("code")["2025M12"]
r2 = reconstruct.levels_new_era(sub_new, w_dec25)
r3 = reconstruct.mm_from_panel(sub_new, reconstruct.DIV_NEW)
r3 = r3[r3.index >= "2026-01"]

# levels: subindices stored at 1dp (+-0.05) -> weighted worst case 0.05 pts
print(f"levels:  max |error| {r2.error.abs().max():.4f} index points (1dp input rounding: <=0.05)")
print(f"m/m:     max |error| {r3.error.abs().max():.4f} pp")
gate3 = r2.error.abs().max() <= 0.05 and r3.error.abs().max() <= budget
print("GATE 3:", "PASS" if gate3 else "FAIL")
r2.round(3)
'''),
("md", "## Gate 4 — VNV án húsnæðis (independent check of the component mapping)"),
("code", '''\
# ex-housing: drop IS04 entirely, add back IS045 (hiti og rafmagn) per its definition
codes_lh = [c for c in reconstruct.DIV_OLD if c != "IS04"] + ["IS045"]
div = panel_old[panel_old.code.isin(codes_lh)].copy().sort_values(["code", "manudur"])
div["w_pre_raw"] = div["vaegi"] / (1 + div["manadarbreyting"] / 100)
g = div.groupby("manudur").apply(
    lambda d: (d.w_pre_raw / d.w_pre_raw.sum() * d.manadarbreyting).sum(), include_groups=False
).to_frame("recon_mm")
g = g[div.groupby("manudur").size().reindex(g.index) == len(codes_lh)]
g["published_mm"] = head[("CPILH", "change_M")].reindex(g.index)
g["error"] = g.recon_mm - g.published_mm
g = g.dropna().loc["2019":]

# The ex-housing basket is ~72% of the total: renormalizing 2dp-rounded weights
# amplifies rounding noise, especially in utsolur months when clothing swings +-10%.
# Empirical tolerance: <=0.03pp per month and unbiased mean.
print(f"months: {len(g)}   max |error|: {g.error.abs().max():.4f} pp   mean: {g.error.mean():+.5f}")
gate4 = g.error.abs().max() <= 0.03 and abs(g.error.mean()) <= 0.002
print("GATE 4:", "PASS" if gate4 else "FAIL")
g.error.abs().groupby(g.index.year).max().round(4).to_frame("max |error| pp")
'''),
("md", """\
**Negative control** — does this check actually have teeth? Break the mapping on purpose
(the classic mistake: dropping *hiti og rafmagn* IS045, which the ex-housing index retains)
and the error should explode:
"""),
("code", '''\
codes_bad = [c for c in reconstruct.DIV_OLD if c != "IS04"]  # WRONG: no IS045
bad = panel_old[panel_old.code.isin(codes_bad)].copy().sort_values(["code", "manudur"])
bad["w_pre_raw"] = bad["vaegi"] / (1 + bad["manadarbreyting"] / 100)
gb = bad.groupby("manudur").apply(
    lambda d: (d.w_pre_raw / d.w_pre_raw.sum() * d.manadarbreyting).sum(), include_groups=False
).to_frame("recon_mm")
gb["error"] = gb.recon_mm - head[("CPILH", "change_M")].reindex(gb.index)
gb = gb.dropna().loc["2019":]
print(f"wrong mapping:   max |error| {gb.error.abs().max():.4f} pp, "
      f"mean {gb.error.mean():+.4f} pp  <- systematic bias, immediately visible")
print(f"correct mapping: max |error| {g.error.abs().max():.4f} pp, mean {g.error.mean():+.5f} pp")
assert gb.error.abs().max() > 3 * g.error.abs().max(), "negative control failed to fail!"
'''),
("md", "## Gate 5 — level chain: no cumulative drift beyond rounding"),
("code", '''\
idx = head[("CPI", "index")]
anchor = pd.Period("2018-12", "M")
lvl = reconstruct.levels_from_mm(r1.recon_mm, anchor, idx[anchor])
lvl_err = (lvl - idx.reindex(lvl.index)).dropna()

# published m/m is rounded to 2dp -> per-month level noise ~0.005% of ~600 = 0.03 pts;
# random-walk accumulation over 84 months stays within ~0.15 pts if unbiased
print(f"7-year chained level drift: max |err| {lvl_err.abs().max():.3f} index points "
      f"on a ~690 index ({lvl_err.abs().max() / idx.iloc[-1] * 100:.3f}%)")
gate5 = lvl_err.abs().max() < 0.15
print("GATE 5:", "PASS" if gate5 else "FAIL")

fig, ax = plt.subplots()
ax.plot(lvl_err.index.to_timestamp(), lvl_err, color=C["blue"], lw=1.5)
ax.set_title("Chained-level drift vs published index (index points), anchored Dec 2018")
plt.show()
'''),
("code", '''\
summary = pd.DataFrame({
    "gate": ["1: m/m identity 2019-2025", "2: chain-link months", "3: COICOP2018 era",
             "4: VNV án húsnæðis", "5: level chain drift"],
    "result": ["PASS" if x else "FAIL" for x in [gate1, gate2, gate3, gate4, gate5]],
}).set_index("gate")
assert (summary.result == "PASS").all(), "HARD GATE FAILED - do not proceed to forecasting"
print("Phase 2 hard gate: ALL PASS - forecasting layer is built on this engine.")
summary
'''),
]

# ============================================================================
# Notebook 3 — forecast
# ============================================================================
nb3 = [
("md", """\
# 03 — Spá (the forecast)

**What this is.** A bottom-up 12-month forecast of VNV: each of 23 components gets a
transparent m/m model, and the paths are aggregated with the **latest published weights,
price-updated through the horizon** by the engine verified in notebook 02.

**Model per component (v0 — deliberately simple and inspectable)**
- *Default*: estimated monthly seasonal factors + AR(1) on the deseasonalized m/m
  (sample 2016–, spliced COICOP2018 history).
- *Reiknuð húsaleiga (CP042)*: post-June-2024 sample only (methodology break), persistence
  model — rents under rental equivalence are slow-moving stock rents.
- Divisions restructured by COICOP2018 (07–10, 12, 13) borrow seasonal/AR parameters from
  their old-classification analogues, **for parameter fitting only** — flagged in
  `models.OLD_PROXY`.

**What v0 does *not* yet include** (Phases 3–6 of the plan): scraped airfares/fuel/grocery
nowcasts, the fiscal & wage calendars, HMS rent pass-through, top-down reconciliation
(MinT). Those mostly sharpen h=1–3. The backtest below is honest about the consequence.
"""),
("code", BOOTSTRAP),
("code", '''\
from vnv import ingest, models

head = ingest.load_headline()
sub_new = ingest.load_sub_new()
panel_old = ingest.load_panel_old()
spliced = ingest.load_sub_spliced()
vt = ingest.load_verdtrygging()

g = models.build_component_history(spliced, panel_old, sub_new)
last_m = g.index.max()
latest = sub_new[sub_new.manudur == last_m].set_index("code")
w0 = latest.loc[models.COMPONENTS, "vaegi"]
print(f"jump-off: {last_m}   component weight sum: {w0.sum():.2f}")
'''),
("md", "## Component models — every parameter visible"),
("code", '''\
fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
param_tbl = pd.DataFrame({
    "heiti": latest.loc[models.COMPONENTS, "heiti"].str.slice(0, 44),
    "vægi (%)": w0,
    "AR(1) φ": [fits[c].phi for c in models.COMPONENTS],
    "resid SD (pp)": [fits[c].resid.std() for c in models.COMPONENTS],
    "n obs": [fits[c].n_obs for c in models.COMPONENTS],
    "note": [fits[c].note for c in models.COMPONENTS],
})
param_tbl.round(3)
'''),
("code", '''\
# Seasonal factors per component (pp per month) - the deterministic core of the forecast
seas_tbl = pd.DataFrame({c: fits[c].seasonal for c in models.COMPONENTS}).T
seas_tbl.columns = ["jan", "feb", "mar", "apr", "maí", "jún", "júl", "ágú", "sep", "okt", "nóv", "des"]
seas_tbl.round(2)
'''),
("md", """\
## The 12-month path

The aggregation below is the answer to *“show the weights used to predict the CPI”*:
each month's headline forecast is exactly `Σ weight × component m/m`, with the weight
path shown explicitly.
"""),
("code", '''\
fc = pd.DataFrame({c: models.forecast_component(fits[c], last_m, 12) for c in models.COMPONENTS})
head_mm, contribs, wpath = models.aggregate_bottom_up(fc, w0)

path = pd.DataFrame({"headline m/m (%)": head_mm})
path["index (spá)"] = head[("CPI", "index")].iloc[-1] * (1 + head_mm / 100).cumprod()
idx_hist = head[("CPI", "index")]
path["y/y (%)"] = (path["index (spá)"] / idx_hist.reindex(path.index - 12).to_numpy() - 1) * 100
print(f"implied 12m inflation to {path.index[-1]}: "
      f"{((1 + head_mm / 100).prod() - 1) * 100:.2f}%")
path.round(2)
'''),
("code", '''\
# Contribution decomposition: weight x m/m = contribution, month by month (pp)
name_map = latest.loc[models.COMPONENTS, "heiti"].str.slice(0, 40)
contrib_show = contribs.T.set_axis(name_map.reindex(contribs.columns), axis=0)
contrib_show["12m samtals"] = contrib_show.sum(axis=1)
contrib_show.sort_values("12m samtals", ascending=False).round(3)
'''),
("code", '''\
# The weight path used at each forecast month (price-updated from the latest published Vaegi)
wpath.T.set_axis(name_map.reindex(wpath.columns), axis=0).round(2)
'''),
("code", '''\
first, last12 = contribs.index[0], contribs.sum()
d = last12.sort_values()
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.barh(name_map.reindex(d.index), d, color=[C["red"] if v < 0 else C["blue"] for v in d], height=0.62)
for i, v in enumerate(d):
    ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8.5)
ax.axvline(0, color=C["black"], lw=0.8)
ax.set_title(f"Framlag til 12 mánaða verðbólgu {path.index[0]}–{path.index[-1]} (pp)")
ax.grid(axis="y", alpha=0)
plt.tight_layout(); plt.show()
'''),
("md", "## Uncertainty — bootstrap fan chart (joint residual months, cross-correlation preserved)"),
("code", '''\
sims = models.bootstrap_paths(fits, fc, w0, n_sims=2000)
idx0 = float(idx_hist.iloc[-1])
lvl_sims = idx0 * (1 + sims / 100).cumprod(axis=1)
base = idx_hist.reindex(sims.columns - 12).to_numpy()
yy_sims = (lvl_sims / base - 1) * 100

qs = yy_sims.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
hist_yy = head[("CPI", "change_A")]["2022":]
fig, ax = plt.subplots()
ax.plot(hist_yy.index.to_timestamp(), hist_yy, color=C["black"], lw=1.8, label="raun y/y")
x = qs.index.to_timestamp()
ax.fill_between(x, qs[0.05], qs[0.95], color=C["sky"], alpha=0.25, lw=0, label="90% bil")
ax.fill_between(x, qs[0.25], qs[0.75], color=C["sky"], alpha=0.45, lw=0, label="50% bil")
ax.plot(x, qs[0.5], color=C["blue"], lw=2, label="spá (miðgildi)")
ax.plot(path.index.to_timestamp(), path["y/y (%)"], color=C["orange"], lw=1.4, ls="--", label="punktspá")
ax.axhline(2.5, color=C["black"], lw=1, ls=":", alpha=0.6)
ax.set_title("Verðbólguspá — y/y með óvissubili (bootstrap)")
ax.legend(frameon=False, ncols=3)
plt.show()
'''),
("md", """\
## Verðtrygging path

VNV published for month $t$ sets the indexation base for month $t+2$; the fund's indexed
instruments settle on `VIS01004`. The first two rows below are therefore already **known**
from published prints — the model only matters from the third row on.
"""),
("code", '''\
# empirical t+2 mapping check on published data
vt_s = vt["financial_indexation"].dropna()
cpi_s = idx_hist.dropna()
check = pd.DataFrame({"verðtrygging(t+2)": vt_s.reindex(cpi_s.index + 2).to_numpy(),
                      "VNV(t)": cpi_s.to_numpy()}, index=cpi_s.index).dropna().tail(6)
check["hlutfall"] = check.iloc[:, 0] / check.iloc[:, 1]
assert (check.hlutfall.round(4) == 1).all(), "t+2 mapping does not hold"
print("verified: verðtrygging index in t+2 equals published VNV of month t\\n")

vt_path = pd.DataFrame({"VNV": path["index (spá)"].round(1)})
vt_path.index = path.index + 2
vt_path.index.name = "verðtryggingarmánuður"
vt_path["m/m verðbóta (%)"] = (vt_path.VNV / pd.concat([idx_hist.tail(2), path["index (spá)"]]).round(1).shift(1).reindex(path.index).to_numpy() - 1) * 100
vt_path.round(2)
'''),
("md", """\
## Honest backtest — walk-forward h=1, model vs floors

Refitted each month on data through $t-1$, aggregated with the *published* weights of
$t-1$ (no look-ahead). Actual = the published headline m/m from `VIS01000`.
"""),
("code", '''\
test_months = [m for m in g.index if m > pd.Period("2025-01", "M") and m <= last_m]
rows = []
for m in test_months:
    g_tr = g[g.index < m]
    w_prev = sub_new[(sub_new.manudur == m - 1) & sub_new.code.isin(models.COMPONENTS)]
    w_prev = w_prev.set_index("code").vaegi
    f = {c: models.forecast_component(models.fit_component(g_tr[c], c), m - 1, 1).iloc[0]
         for c in models.COMPONENTS}
    hm, _, _ = models.aggregate_bottom_up(pd.DataFrame([f], index=[m]), w_prev)
    seas = {c: g_tr[c][(g_tr.index.month == m.month) & (g_tr.index >= "2016-01")].mean()
            for c in models.COMPONENTS}
    hs, _, _ = models.aggregate_bottom_up(pd.DataFrame([seas], index=[m]), w_prev)
    rows.append({"manudur": m, "model": hm.iloc[0], "seasonal naive": hs.iloc[0],
                 "random walk": float(g.loc[m - 1] @ (w_prev / w_prev.sum())),
                 "actual": head[("CPI", "change_M")].get(m, np.nan)})
bt = pd.DataFrame(rows).set_index("manudur").dropna()

stats = pd.DataFrame({
    c: {"RMSE (pp)": np.sqrt(((bt[c] - bt.actual) ** 2).mean()),
        "MAE (pp)": (bt[c] - bt.actual).abs().mean()}
    for c in ["model", "seasonal naive", "random walk"]
}).T
display(bt.round(3))
stats.round(3)
'''),
("md", """\
**Read this honestly.** Over the 18-month window the v0 model roughly matches the
seasonal-naive floor and clearly beats the random walk — it does **not** yet add value
at h=1, exactly as `PLAN_1.md` anticipates: the h=1 misses are dominated by airfares,
útsölur (Jan/Feb sales) and one-off administered steps, which are *observable* (Phase 3
scrapers, fiscal calendar), not forecastable from history. Do not tune this away.

The economic use case is h=3–12, where the fan chart above is compared against breakeven
inflation (RIKS vs RIKB) — that comparison needs Seðlabanki market data (Phase 6/7).

### Next steps per the plan
1. **Phase 3**: airfare & fuel scrapers matched to the collection window; fiscal calendar.
2. **Phase 4**: HMS leiguvísitala → CP042 distributed-lag map (vintage-aware).
3. **Phase 5**: FX/wage ARDL blocks; wage calendar as dated steps.
4. **Phase 6**: BVAR top-down + MinT reconciliation; fan chart from reconciled covariance.
5. **Phase 7**: analyst/Seðlabanki/breakeven benchmark table.
"""),
]

# ============================================================================
# Notebook 4 — observable-input nowcast (Phase 3)
# ============================================================================
nb4 = [
("md", """\
# 04 — Mælanlegt í stað spáðs (the observable-input nowcast, Phase 3)

By mid-month a share of the basket is **observable rather than forecastable**. This
notebook wires in the first observable — fuel — and shows exactly why observables and
the fiscal calendar must arrive together.

| Input | Source | Status |
|---|---|---|
| Eldsneyti (CP0722, ~3.5%) | Gasvaktin open data — every retail price change since 2016 | **live, calibrated** |
| Flugfargjöld (CP07332, ~2.5%) | Icelandair/PLAY quotes | framework only (`airfares.py`) — needs a collection path |
| Matvörur (CP011x, ~13.7%) | Krónan mirror (home_app Supabase) | pipeline ready (`groceries.py`) — history starts when snapshots begin |
| Administered steps | `config/fiscal_calendar.yaml` | scaffolded; Jan-2026 steps realized & quantified |

No placeholder data anywhere: pending sources return empty rather than synthetic numbers.
"""),
("code", BOOTSTRAP),
("code", '''\
from vnv import ingest, models, nowcast, fuel

head = ingest.load_headline()
sub_new = ingest.load_sub_new()
panel_old = ingest.load_panel_old()
spliced = ingest.load_sub_spliced()
g = models.build_component_history(spliced, panel_old, sub_new)
last_m = g.index.max()
'''),
("md", """\
## Fuel: ten years of pump prices vs the published subindex

Gasvaktin logs every verified pump-price change per retailer. We average list prices
across retailers (Costco excluded — not in Hagstofa's outlet set), take the mean over
the collection window (days 1–15), and regress the published subindex m/m on it.
"""),
("code", '''\
bensin = fuel.scraped_mm("bensin95")
diesel = fuel.scraped_mm("diesel")
mix = (2/3) * bensin + (1/3) * diesel
pub_e = panel_old[panel_old.code == "IS0722"].set_index("manudur").manadarbreyting

cal = nowcast.calibrate_fuel()
print(f"calibration (2016–2025, ex 2026-01/02): n={cal.n_obs}   "
      f"published = {cal.alpha:+.3f} + {cal.beta:.3f} × scraped   resid SD = {cal.resid_sd:.2f}pp")
print(f"error budget at ~3.5% weight: {cal.resid_sd * 0.035:.3f}pp on the headline")

df = pd.concat([mix.rename("scraped"), pub_e.rename("published")], axis=1).dropna()
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(df.scraped, df.published, s=18, color=C["blue"], alpha=0.6)
lims = np.array([df.min().min() - 1, df.max().max() + 1])
ax.plot(lims, cal.alpha + cal.beta * lims, color=C["orange"], lw=2)
jan = df.loc[[pd.Period("2026-01", "M")]] if pd.Period("2026-01", "M") in df.index else None
if jan is not None:
    ax.scatter(jan.scraped, jan.published, s=60, color=C["red"], zorder=5)
    ax.annotate("jan 2026\\n(skattabreyting)", xy=(jan.scraped.iloc[0], jan.published.iloc[0]),
                xytext=(8, 8), textcoords="offset points", fontsize=9, color=C["red"])
ax.set_xlabel("scraped: söfnunarglugga-meðaltal, m/m %")
ax.set_ylabel("published: IS0722 eldsneyti, m/m %")
ax.set_title("Dæluverð (Gasvaktin) spannar eldsneytisliðinn — 2016–2025")
plt.show()
'''),
("md", """\
## The January-2026 lesson: observables and the fiscal calendar are a pair

Backtest h=1 with and without the fuel observable. In normal months the observable
helps. In January 2026 it makes the forecast **worse** — not because the fuel reading
was wrong (it nailed the −26% fall, áhrif −0.94pp) but because the same reform package
introduced the kílómetragjald (CP0724: **+134% m/m, +1.01pp**) and cut the EV subsidy
(CP071: +12.3%, +0.55pp), which the model had no way to see. Observing one leg of a
fiscal package without booking the others is worse than observing neither.
Those steps now live in `config/fiscal_calendar.yaml`.
"""),
("code", '''\
rows = []
for m in [m for m in g.index if m > pd.Period("2025-01", "M")]:
    g_tr = g[g.index < m]
    w_prev = sub_new[(sub_new.manudur == m - 1) & sub_new.code.isin(models.COMPONENTS)]
    w_prev = w_prev.set_index("code").vaegi
    if len(w_prev) < len(models.COMPONENTS):
        continue
    f = {c: models.forecast_component(models.fit_component(g_tr[c], c), m - 1, 1).iloc[0]
         for c in models.COMPONENTS}
    fmm = pd.DataFrame([f], index=[m])
    hm, _, _ = models.aggregate_bottom_up(fmm, w_prev)
    cal_m = nowcast.calibrate_fuel(train_end=m - 1)
    fmm_o = nowcast.apply_observables(fmm, nowcast.fuel_nowcast(m, cal_m))
    hmo, _, _ = models.aggregate_bottom_up(fmm_o, w_prev)
    rows.append({"manudur": m, "model": hm.iloc[0], "model+fuel": hmo.iloc[0],
                 "actual": head[("CPI", "change_M")].get(m, np.nan)})
bt = pd.DataFrame(rows).set_index("manudur").dropna()

def rmse(c, frame):
    return np.sqrt(((frame[c] - frame.actual) ** 2).mean())

ex_jan = bt.drop(pd.Period("2026-01", "M"))
stats = pd.DataFrame({
    "RMSE all": {c: rmse(c, bt) for c in ["model", "model+fuel"]},
    "RMSE ex jan-2026": {c: rmse(c, ex_jan) for c in ["model", "model+fuel"]},
})
display(bt.round(3))
stats.round(3)
'''),
("md", "## Today's nowcast (partial collection window)"),
("code", '''\
target = last_m + 1
fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
fc1 = pd.DataFrame({c: models.forecast_component(fits[c], last_m, 1) for c in models.COMPONENTS})
obs = nowcast.fuel_nowcast(target, cal)
fc1_obs = nowcast.apply_observables(fc1, obs)

latest = sub_new[sub_new.manudur == last_m].set_index("code")
w0 = latest.loc[models.COMPONENTS, "vaegi"]
hm, contribs, _ = models.aggregate_bottom_up(fc1_obs, w0)

tbl = pd.DataFrame({
    "heiti": latest.loc[models.COMPONENTS, "heiti"].str.slice(0, 40),
    "vægi (%)": w0.round(2),
    "m/m spá (%)": fc1_obs.iloc[0].round(2),
    "framlag (pp)": contribs.iloc[0].round(3),
    "uppruni": ["mælt (Gasvaktin)" if c in obs else "líkan" for c in models.COMPONENTS],
})
print(f"h=1 nowcast for {target}: {hm.iloc[0]:+.2f}% m/m "
      f"(fuel observed at {obs['CP0722']:+.2f}%)")
print("NB: collection window still open — fuel reading updates daily until the 15th.")
tbl.sort_values("framlag (pp)", ascending=False)
'''),
("md", """\
## Pending observables — status, not placeholders

**Groceries (CP011x):** the Krónan catalog mirror (8,295 SKUs, built for home_app) holds
current prices only. History accumulates via daily snapshots; `groceries.py` then builds
a matched-model Jevons index per COICOP class with the category mapping already defined.
Two feed options (owner decision): the server-side snapshot table + pg_cron
(`scripts/kronan_history_migration.sql`, not applied) or the local scraper
(`scripts/snapshot_kronan.py`, needs `KRONAN_API_TOKEN`).

**Airfares (CP07332):** the single highest-variance input. `airfares.py` defines the
quote-basket design (routes × booking offsets × collection window) and the index math,
but collecting quotes needs either a fares API key or a headless-browser scraper —
listed as the top follow-up. Until then CP073 stays model-driven, which is the dominant
term in the h=1 error budget.
"""),
("code", '''\
from vnv import groceries, airfares
print("grocery snapshots on disk:", len(list(groceries.SNAP_DIR.glob('snapshot_*.csv'))
      if groceries.SNAP_DIR.exists() else []))
print("grocery food index:", "EMPTY - awaiting snapshots" if groceries.food_index_mm().empty
      else "data available")
print("airfare quotes on disk:", len(airfares.load_quotes()), "-> index:",
      "EMPTY - awaiting quotes" if airfares.airfare_index_mm().empty else "data available")

import yaml
cal_steps = yaml.safe_load(open(ROOT / "config" / "fiscal_calendar.yaml", encoding="utf-8"))
pd.DataFrame(cal_steps)[["date", "name", "affects", "direction", "status"]]
'''),
("md", """\
### Error budget after Phase 3 so far

| Component | h=1 error source | SD (pp on headline) |
|---|---|---|
| Eldsneyti | calibration residual 0.86pp × ~3.5% | **0.03** (was ~0.10 unobserved) |
| Flugfargjöld/CP073 | unobserved, ±10–20% swings × ~3.9% | ~0.25 — **the binding constraint** |
| Matur | model-only until snapshots accumulate | ~0.08 |
| Admin steps | calendar scaffolded, quantification manual | episodic (±1pp in January) |

The h=1 target of ≤0.15pp RMSE (Phase 5 gate) is not reachable while airfares are
unobserved — that is the next data source to secure, exactly as the plan ranks it.
"""),
]

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in which:
        build(NB_DIR / "01_data_and_weights.ipynb", nb1)
    if "2" in which:
        build(NB_DIR / "02_reconstruction_check.ipynb", nb2)
    if "3" in which:
        build(NB_DIR / "03_forecast.ipynb", nb3)
    if "4" in which:
        build(NB_DIR / "04_nowcast.ipynb", nb4)
