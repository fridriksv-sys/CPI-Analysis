"""VNV forecast dashboard.

Run:  .venv\\Scripts\\python.exe -m streamlit run streamlit_app.py

Everything shown here is computed by the same src/vnv modules the notebooks
use; the notebooks remain the audit trail, this is the daily view.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from vnv import airfares, fuel, groceries, ingest, models, nowcast, reconstruct, rent

st.set_page_config(page_title="VNV spá", page_icon="📈", layout="wide")

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "sky": "#56B4E9", "black": "#222222"}
plt.rcParams.update({
    "figure.figsize": (10, 4), "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


# ---------------------------------------------------------------- data layer
@st.cache_data(ttl=3600, show_spinner="Sæki gögn frá Hagstofunni ...")
def load_all():
    head = ingest.load_headline()
    sub_new = ingest.load_sub_new()
    panel_old = ingest.load_panel_old()
    spliced = ingest.load_sub_spliced()
    g = models.build_component_history(spliced, panel_old, sub_new)
    return head, sub_new, panel_old, g


@st.cache_data(ttl=3600, show_spinner="Reikna spá ...")
def run_forecast():
    head, sub_new, panel_old, g = load_all()
    last_m = g.index.max()
    latest = sub_new[sub_new.manudur == last_m].set_index("code")
    w0 = latest.loc[models.COMPONENTS, "vaegi"]
    fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
    spl = ingest.load_sub_spliced()
    fc = models.forecast_components(fits, last_m, 12, hms_rent_mm=rent.hms_rent_mm(),
                                    sub_spliced=spl)
    cal = nowcast.calibrate_fuel()
    try:
        obs = nowcast.fuel_nowcast(last_m + 1, cal)
    except ValueError:
        obs = {}
    fc_obs = nowcast.apply_observables(fc, obs)
    head_mm, contribs, wpath = models.aggregate_bottom_up(fc_obs, w0)
    sims = models.bootstrap_paths(fits, fc_obs, w0, n_sims=1000)
    return dict(last_m=last_m, latest=latest, w0=w0, fc=fc_obs, obs=obs, cal=cal,
                head_mm=head_mm, contribs=contribs, wpath=wpath, sims=sims)


head, sub_new, panel_old, g = load_all()
R = run_forecast()
last_m, latest, w0 = R["last_m"], R["latest"], R["w0"]
idx_hist = head[("CPI", "index")]
name_map = latest.loc[models.COMPONENTS, "heiti"].str.slice(0, 40)

st.title("Vísitala neysluverðs — spálíkan")
st.caption(f"Nýjasta birting: {last_m} · VNV {idx_hist.iloc[-1]:.1f} · "
           f"12M verðbólga {head[('CPI', 'change_A')].iloc[-1]:.1f}%")

tab_now, tab_fc, tab_w, tab_gate, tab_feed = st.tabs(
    ["Núspá", "12 mánaða spá", "Vogir", "Endurbygging (gátt)", "Gagnalindir"])

# ---------------------------------------------------------------- tab: nowcast
with tab_now:
    target = last_m + 1
    hm1 = R["head_mm"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Núspá {target} (m/m)", f"{hm1:+.2f}%")
    yy_impl = (idx_hist.iloc[-1] * (1 + hm1 / 100) / idx_hist.loc[target - 12] - 1) * 100
    c2.metric("12M verðbólga við birtingu", f"{yy_impl:.1f}%")
    if R["obs"]:
        c3.metric("Eldsneyti (mælt, Gasvaktin)", f"{R['obs']['CP0722']:+.2f}%",
                  help="Kvarðað dæluverð yfir söfnunarglugga (1.–15.); uppfærist daglega")
    st.caption("Söfnunargluggi Hagstofunnar er 1.–15. — núspáin batnar fram að því.")

    tbl = pd.DataFrame({
        "Liður": name_map,
        "Vægi %": w0.round(2),
        "m/m spá %": R["fc"].iloc[0].round(2),
        "Framlag pp": R["contribs"].iloc[0].round(3),
        "Uppruni": ["mælt" if c in R["obs"] else "líkan" for c in models.COMPONENTS],
    }).sort_values("Framlag pp", ascending=False)
    st.dataframe(tbl, height=420, width="stretch")

    st.subheader("Dæluverð í rauntíma (Gasvaktin)")
    fig, ax = plt.subplots()
    for f_, col, lab in [("bensin95", C["blue"], "bensín 95"), ("diesel", C["orange"], "dísel")]:
        s = fuel.national_price(f_)
        s = s[s.index >= s.index.max() - pd.Timedelta(days=180)]
        ax.plot(s.index, s, color=col, lw=1.8, label=lab)
    ax.set_ylabel("kr/l (listaverð, meðaltal söluaðila)")
    ax.legend(frameon=False)
    st.pyplot(fig, width="content")

# ---------------------------------------------------------------- tab: forecast
with tab_fc:
    path_idx = idx_hist.iloc[-1] * (1 + R["head_mm"] / 100).cumprod()
    yy = (path_idx / idx_hist.reindex(R["head_mm"].index - 12).to_numpy() - 1) * 100
    lvl_sims = idx_hist.iloc[-1] * (1 + R["sims"] / 100).cumprod(axis=1)
    yy_sims = (lvl_sims / idx_hist.reindex(R["sims"].columns - 12).to_numpy() - 1) * 100
    qs = yy_sims.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T

    m1, m2 = st.columns(2)
    m1.metric("Verðbólga eftir 12 mánuði (spá)", f"{yy.iloc[-1]:.1f}%")
    m2.metric("90% bil", f"{qs[0.05].iloc[-1]:.1f}% – {qs[0.95].iloc[-1]:.1f}%")

    hist_yy = head[("CPI", "change_A")]["2022":]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist_yy.index.to_timestamp(), hist_yy, color=C["black"], lw=1.8, label="raun")
    x = qs.index.to_timestamp()
    ax.fill_between(x, qs[0.05], qs[0.95], color=C["sky"], alpha=0.25, lw=0, label="90% bil")
    ax.fill_between(x, qs[0.25], qs[0.75], color=C["sky"], alpha=0.45, lw=0, label="50% bil")
    ax.plot(x, qs[0.5], color=C["blue"], lw=2, label="spá")
    ax.axhline(2.5, color=C["black"], lw=1, ls=":", alpha=0.6)
    ax.set_title("Verðbólguspá y/y með óvissubili")
    ax.legend(frameon=False, ncols=4)
    st.pyplot(fig, width="stretch")

    st.subheader("Framlög til 12 mánaða verðbólgu (pp)")
    last12 = R["contribs"].sum().sort_values()
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.barh(name_map.reindex(last12.index), last12,
             color=[C["red"] if v < 0 else C["blue"] for v in last12], height=0.6)
    ax2.axvline(0, color=C["black"], lw=0.8)
    ax2.grid(axis="y", alpha=0)
    st.pyplot(fig2, width="stretch")

    st.subheader("Verðtryggingarferill (VNV í mánuði t → verðtrygging í t+2)")
    vt_tbl = pd.DataFrame({
        "VNV (spá)": path_idx.round(1),
        "m/m %": R["head_mm"].round(2),
        "verðtryggingarmánuður": (R["head_mm"].index + 2).astype(str),
    })
    st.dataframe(vt_tbl, width="stretch")

# ---------------------------------------------------------------- tab: weights
with tab_w:
    st.caption("Vægi = verðuppfærð hlutdeild í körfunni (Vægi %, Hagstofan). "
               "Spáin notar nýjustu birtu vogir og verðuppfærir þær eftir spábraut.")
    d = pd.DataFrame({"heiti": name_map, "vægi": w0}).sort_values("vægi")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(d.heiti, d.vægi, color=C["blue"], height=0.6)
    for i, v in enumerate(d.vægi):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_title(f"Vægi spáliða, {last_m} (%)")
    ax.grid(axis="y", alpha=0)
    st.pyplot(fig, width="stretch")

    st.subheader("Vægi reiknaðrar húsaleigu gegnum tíðina")
    w_old = panel_old[panel_old.code == "IS042"].set_index("manudur").vaegi["2015":]
    w_new_s = sub_new[sub_new.code == "CP042"].set_index("manudur").vaegi
    fig2, ax2 = plt.subplots()
    ax2.plot(w_old.index.to_timestamp(), w_old, color=C["blue"], lw=1.8, label="IS042 (eldra kerfi)")
    ax2.plot(w_new_s.index.to_timestamp(), w_new_s, color=C["orange"], lw=1.8, label="CP042 (COICOP2018)")
    ax2.axvline(pd.Period("2024-06", "M").to_timestamp(), color=C["red"], ls="--", lw=1)
    ax2.legend(frameon=False)
    st.pyplot(fig2, width="stretch")

    st.subheader("Vogaferill spár (verðuppfærður)")
    st.dataframe(R["wpath"].T.set_axis(name_map.reindex(R["wpath"].columns), axis=0).round(2),
                 width="stretch")

# ---------------------------------------------------------------- tab: gates
with tab_gate:
    st.caption("Harða gáttin úr PLAN.md fasa 2: birt vísitala endurbyggð úr liðum. "
               "Full úttekt í notebooks/02_reconstruction_check.ipynb.")
    r1 = reconstruct.mm_from_panel(panel_old, reconstruct.DIV_OLD).loc["2019":]
    budget = reconstruct.rounding_budget_mm(pd.Series(100 / 12, index=reconstruct.DIV_OLD), 12)
    r3 = reconstruct.mm_from_panel(sub_new, reconstruct.DIV_NEW).loc["2026":]

    g1 = r1.error.abs().max() <= budget
    g3 = r3.error.abs().max() <= budget
    c1, c2, c3 = st.columns(3)
    c1.metric("2019–2025 (84 mán.)", "PASS ✅" if g1 else "FAIL ❌",
              f"hámarksfrávik {r1.error.abs().max():.4f} pp")
    c2.metric("COICOP2018-tímabil", "PASS ✅" if g3 else "FAIL ❌",
              f"hámarksfrávik {r3.error.abs().max():.4f} pp")
    c3.metric("Námundunarþol", f"{budget:.4f} pp", "birt gögn: 2 aukastafir")

    fig, ax = plt.subplots()
    ax.plot(r1.index.to_timestamp(), r1.error * 100, color=C["blue"], lw=1.4)
    ax.axhspan(-budget * 100, budget * 100, color=C["sky"], alpha=0.25, lw=0)
    ax.set_title("Endurbyggingarfrávik m/m (punktar) og námundunarþol")
    ax.set_ylabel("bp")
    st.pyplot(fig, width="stretch")

# ---------------------------------------------------------------- tab: feeds
with tab_feed:
    fuel_last = fuel.national_price("bensin95").dropna()
    n_snap = len(groceries.load_snapshots())
    n_quotes = len(airfares.load_quotes())
    rows = [
        ("Hagstofan PxWeb", "✅ virk", f"nýjast: {last_m}"),
        ("Eldsneyti — Gasvaktin", "✅ virk",
         f"nýjast: {fuel_last.index[-1]:%Y-%m-%d} ({fuel_last.iloc[-1]:.1f} kr/l bensín)"),
        ("Húsaleiga — HMS leiguvísitala", "✅ virk",
         f"nýjast: {rent.hms_rent_mm().dropna().index[-1]} · knýr CP042 líkanið"),
        ("Matvörur — Krónan (kronan_price_history í Supabase)", "🟡 söfnun hafin",
         f"{n_snap} raðir í staðbundnu afriti — keyra scripts/export_kronan_history.py; "
         "kvörðun þegar saga spannar ≥2 söfnunarglugga"),
        ("Flugfargjöld — Icelandair (KEF, ISK)", "🟡 söfnun hafin",
         f"{n_quotes} fargjöld skráð í dag; dagleg söfnun (1.–15.); "
         "kvörðun þegar saga spannar ≥2 söfnunarglugga"),
        ("Fjármálaalmanak", "🟡 handvirkt", "config/fiscal_calendar.yaml"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Lind", "Staða", "Athugasemd"]),
                 width="stretch", hide_index=True)

    st.subheader("Kvörðun eldsneytis (2016–2025)")
    cal = R["cal"]
    st.write(f"`published = {cal.alpha:+.3f} + {cal.beta:.3f} × scraped` · "
             f"n={cal.n_obs} · residual SD {cal.resid_sd:.2f}pp "
             f"(~{cal.resid_sd * 0.035:.3f}pp á heildarvísitölu)")
    bensin = fuel.scraped_mm("bensin95")
    diesel = fuel.scraped_mm("diesel")
    mix = (2 / 3) * bensin + (1 / 3) * diesel
    pub_e = panel_old[panel_old.code == "IS0722"].set_index("manudur").manadarbreyting
    df = pd.concat([mix.rename("scraped"), pub_e.rename("published")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.scraped, df.published, s=14, color=C["blue"], alpha=0.6)
    lims = np.array([df.min().min() - 1, df.max().max() + 1])
    ax.plot(lims, cal.alpha + cal.beta * lims, color=C["orange"], lw=2)
    ax.set_xlabel("scraped m/m %")
    ax.set_ylabel("published IS0722 m/m %")
    st.pyplot(fig, width="content")
