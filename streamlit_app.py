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

from vnv import airfares, blocks, fuel, groceries, ingest, models, nowcast, reconstruct, rent, sedlabanki

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


@st.cache_data(ttl=3600, show_spinner="Reikna vogaferil ...")
def weight_path():
    """Price-updated weight path (for the Vogir diagnostics view)."""
    head, sub_new, panel_old, g = load_all()
    last_m = g.index.max()
    w0 = sub_new[sub_new.manudur == last_m].set_index("code").loc[models.COMPONENTS, "vaegi"]
    fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
    spl = ingest.load_sub_spliced()
    fc = models.forecast_components(fits, last_m, 12, hms_rent_mm=rent.hms_rent_mm(),
                                    sub_spliced=spl, comp_history=g, fx_mm=sedlabanki.fx_mm())
    _, _, wpath = models.aggregate_bottom_up(fc, w0)
    return wpath


head, sub_new, panel_old, g = load_all()
last_m = g.index.max()
latest = sub_new[sub_new.manudur == last_m].set_index("code")
w0 = latest.loc[models.COMPONENTS, "vaegi"]
idx_hist = head[("CPI", "index")]
name_map = latest.loc[models.COMPONENTS, "heiti"].str.slice(0, 40)

st.title("Vísitala neysluverðs — spálíkan")
st.caption(f"Nýjasta birting: {last_m} · VNV {idx_hist.iloc[-1]:.1f} · "
           f"12M verðbólga {head[('CPI', 'change_A')].iloc[-1]:.1f}%")

tab_rep, tab_diag = st.tabs(["Skýrsla", "Gögn & gæði"])

# ================================================================ tab: diagnostics
# ---- Vogir (weights) ----
with tab_diag:
    st.header("Vogir")
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
    _wp = weight_path()
    st.dataframe(_wp.T.set_axis(name_map.reindex(_wp.columns), axis=0).round(2),
                 width="stretch")

# ---- Endurbygging (reconstruction gate) ----
with tab_diag:
    st.divider()
    st.header("Endurbygging (gátt)")
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

# ---- Gagnalindir (data feeds) ----
with tab_diag:
    st.divider()
    st.header("Gagnalindir")
    fuel_last = fuel.national_price("bensin95").dropna()
    n_snap = len(groceries.load_snapshots())
    n_quotes = len(airfares.load_quotes())
    from vnv import benchmarks as _bm
    n_analyst = len(_bm.load_analyst_forecasts())
    n_pm = len(_bm.load_peningamal())
    rows = [
        ("Hagstofan PxWeb", "✅ virk", f"nýjast: {last_m}"),
        ("Eldsneyti — Gasvaktin", "✅ virk",
         f"nýjast: {fuel_last.index[-1]:%Y-%m-%d} ({fuel_last.iloc[-1]:.1f} kr/l bensín)"),
        ("Húsaleiga — HMS leiguvísitala", "✅ virk",
         f"nýjast: {rent.hms_rent_mm().dropna().index[-1]} · knýr CP042 líkanið"),
        ("Gengi — Seðlabanki gengisvísitala", "✅ virk",
         f"nýjast: {sedlabanki.load_fx().index[-1]} · FX-áhrif á CP01/CP02/CP03/CP071"),
        ("Laun — Hagstofa launavísitala", "✅ virk",
         f"nýjast: {ingest.wages_mm().dropna().index[-1]} · kjarasamningar í wage_calendar.yaml"),
        ("Heimsmatvöruverð — FAO", "✅ virk", "prófað fyrir CP01 — of veikt, sleppt"),
        ("Verðbólguálag (RIKB/RIKS) — lanamal.is", "✅ virk",
         "LoadChartData API; keyra lanamal.update_breakeven_slot() til uppfærslu"),
        ("Greiningaraðilar (Íslandsbanki/Landsbankinn)", "🟡 handvirkt",
         f"{n_analyst} spár skráðar; bæta við hvern mánuð í analyst_forecasts.csv"),
        ("Seðlabanki Peningamál (y/y spá)", "🟡 handvirkt",
         f"{n_pm} ársfj. skráðir; uppfæra ~4x/ári úr Peningamál Töflu 5"),
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

    st.subheader("Dæluverð í rauntíma (Gasvaktin)")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for f_, col, lab in [("bensin95", C["blue"], "bensín 95"), ("diesel", C["orange"], "dísel")]:
        s = fuel.national_price(f_)
        s = s[s.index >= s.index.max() - pd.Timedelta(days=180)]
        ax.plot(s.index, s, color=col, lw=1.8, label=lab)
    ax.set_ylabel("kr/l (listaverð, meðaltal söluaðila)")
    ax.legend(frameon=False)
    st.pyplot(fig, width="content")

    st.subheader("Kvörðun eldsneytis (2016–2025)")
    cal = nowcast.calibrate_fuel()
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

# ================================================================ tab: report (main)
with tab_rep:
    from vnv import report as _report

    @st.cache_data(ttl=3600, show_spinner="Bý til mánaðarskýrslu ...")
    def _monthly():
        try:  # refresh market breakeven (RIKB/RIKS) from lanamal.is; keep last on failure
            from vnv import lanamal
            lanamal.update_breakeven_slot(use_cache=False)
        except Exception:
            pass
        return _report.build_report()

    rep = _monthly()

    # --- 1) Predicted next month's CPI -------------------------------------
    st.header(f"1 · Næsti mánuður — {rep['nowcast_month']}")
    nm = rep["nowcast_month"]
    next_index = idx_hist.iloc[-1] * (1 + rep["nowcast_mm"] / 100)
    yy_next = (next_index / idx_hist.loc[nm - 12] - 1) * 100
    a, b, c = st.columns(3)
    a.metric(f"Núspá m/m ({nm})", f"{rep['nowcast_mm']:+.2f}%")
    b.metric("VNV vísitala (spá)", f"{next_index:.1f}", f"{rep['nowcast_mm']:+.2f}%")
    c.metric("12M verðbólga við birtingu", f"{yy_next:.1f}%")
    st.caption("Núspá = reconciled (MinT) h=1. Söfnunargluggi Hagstofunnar er 1.–15.; "
               "eldsneyti er þegar mælt, önnur mæld gögn bætast við fram að birtingu.")

    # --- 2) Trend over the forecast horizon (36 months) -------------------
    st.header("2 · Leitni — næstu 3 ár")
    lo, hi = rep["band90"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Verðbólga eftir 12 mánuði", f"{rep['yy_12m']:.1f}%")
    m2.metric("90% óvissubil (12m, MinT)", f"{lo:.1f}% – {hi:.1f}%")
    m3.metric(f"Eftir 36 mán. ({rep['yy_path'].index[-1]})", f"{rep['yy_path'].iloc[-1]:.1f}%")
    yy_path = rep["yy_path"]; cum_sd = np.sqrt((rep["rec_sd"] ** 2).cumsum())
    hist_yy = head[("CPI", "change_A")]["2022":]
    x = yy_path.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_yy.index.to_timestamp(), hist_yy, color=C["black"], lw=1.8, label="raun")
    ax.fill_between(x, yy_path - 1.64 * cum_sd, yy_path + 1.64 * cum_sd, color=C["sky"], alpha=0.25, lw=0, label="90% bil")
    ax.fill_between(x, yy_path - 0.67 * cum_sd, yy_path + 0.67 * cum_sd, color=C["sky"], alpha=0.45, lw=0, label="50% bil")
    ax.plot(x, yy_path, color=C["blue"], lw=2.2, label="spá (reconciled)")
    ax.axhline(2.5, color=C["black"], lw=1, ls=":", alpha=0.6)
    ax.set_title("Verðbólguspá y/y (%) — 36 mánaða braut")
    ax.legend(frameon=False, ncols=4)
    st.pyplot(fig, width="stretch")
    st.caption("Fyrstu ~12 mánuðir bera botn-upp smáatriði (húsaleiga, gengi, eldsneyti); "
               "eftir það ræður top-down akkerið (glóðar að 2,5% markmiði) og óvissan breikkar. "
               "/ Bottom-up detail drives the first ~12m; beyond that the top-down anchor "
               "(glide to the 2.5% target) governs and the bands widen.")

    with st.expander("Verðtryggingarferill (VNV í mánuði t → verðtrygging í t+2)"):
        st.dataframe(rep["verdtrygging"].reset_index(), width="stretch", hide_index=True)

    # --- 3) Each underlying: trend + how it was derived -------------------
    st.header("3 · Undirliðir — leitni og aðferð")
    st.caption("Hver liður sem drífur spána: söguleg vísitala (blátt) og 12-mánaða "
               "spá (appelsínugult, brotalína), framlag til verðbólgu og lýsing á aðferð.")
    details = rep["details"]
    cur_block = None
    for det in details:
        if det["block"] != cur_block:
            cur_block = det["block"]
            lab_is, lab_en = _report.BLOCK_LABELS.get(cur_block, (cur_block, cur_block))
            st.subheader(f"{lab_is} / {lab_en}")
        title = (f"{det['code']} · {det['heiti'][:46]}  —  vægi {det['weight']:.1f}% · "
                 f"framlag 12m {det['contrib12']:+.2f}pp · næsti mán. {det['mm_next']:+.2f}%")
        with st.expander(title):
            cc1, cc2 = st.columns([3, 2])
            with cc1:
                hi_, fi_ = det["hist_index"], det["fcst_index"]
                figc, axc = plt.subplots(figsize=(5.4, 2.6))
                if len(hi_):
                    axc.plot(hi_.index.to_timestamp(), hi_.values, color=C["blue"], lw=1.8, label="raun")
                    # connect last actual to the forecast
                    fx_idx = [hi_.index[-1]] + list(fi_.index)
                    fx_val = [hi_.iloc[-1]] + list(fi_.values)
                    axc.plot([p.to_timestamp() for p in fx_idx], fx_val,
                             color=C["orange"], lw=1.8, ls="--", label="spá")
                axc.set_title(f"{det['code']} vísitala", fontsize=9)
                axc.legend(frameon=False, fontsize=8)
                axc.tick_params(labelsize=7)
                st.pyplot(figc, width="stretch")
            with cc2:
                st.markdown(f"**Aðferð / Method:** {det['method_label']}")
                st.caption(det["method_desc"])

    # --- 4) Model vs market breakeven -------------------------------------
    st.header("4 · Módel vs markaður (verðbólguálag)")
    mvb = rep["model_vs_breakeven"]
    if mvb is None:
        st.info("Breakeven-gögn óuppfyllt. Keyra `lanamal.update_breakeven_slot()` "
                "til að sækja RIKB/RIKS af lanamal.is.")
    else:
        cc = mvb["curve"]; term = mvb["model_term"]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Módel {mvb['model_horizon_yrs']:.0f}-ára meðalverðbólga",
                  f"{mvb['model_avg']:.2f}%")
        k2.metric(f"Markaðsálag (~{mvb['breakeven_horizon_yrs']:.0f}á)",
                  f"{mvb['breakeven']:.2f}%")
        k3.metric("Fleygur (álag − módel)", f"{mvb['implied_wedge']:+.2f}pp")
        hs = [r["horizon_yrs"] for r in cc]; be = [r["breakeven_pct"] for r in cc]
        mh = [t["horizon_yrs"] for t in term]; mv = [t["avg_infl"] for t in term]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(hs, be, color=C["green"], lw=2, marker="o", label="markaðsálag (RIKB−RIKS)")
        ax.plot(mh, mv, color=C["blue"], lw=2, marker="D", ms=7, label="módel meðalverðbólga (1/2/3 ár)")
        ax.axhline(2.5, color=C["black"], lw=1, ls=":", alpha=0.6, label="markmið 2,5%")
        ax.set_xlabel("sjóndeildarhringur (ár)"); ax.set_ylabel("meðal-ársverðbólga %")
        ax.set_title("Verðbólga eftir sjóndeildarhring: módel vs markaður")
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, width="stretch")
        st.caption("Nú samræmdir sjóndeildarhringar: T-ára álag ≈ meðalverðbólga markaðar "
                   "næstu T ár, borið saman við T-ára meðalverðbólgu módelsins (úr 36-mán. "
                   "brautinni). Álag umfram módel = áhættu- + skortsálag verðtryggðra bréfa — "
                   "þar liggur h=3–12 forskotið. / Horizon-matched: a T-year breakeven ≈ the "
                   "market's average inflation over T years, vs the model's T-year average. "
                   "Breakeven above the model is the risk + scarcity premium.")

    # --- 5) Model vs bank analysts ---------------------------------------
    st.header("5 · Módel vs greiningaraðilar")
    an = rep.get("analysts")
    if not an:
        st.info("Engar greiningarspár skráðar. Bæta við í data/benchmarks/analyst_forecasts.csv "
                "(bankaspár birtast ~viku fyrir hvern print).")
    else:
        cur = an["current"]
        if cur:
            st.markdown(f"**Næsti print {cur['month']} — spár (m/m):**")
            cols = st.columns(len(cur["analysts"]) + 2)
            cols[0].metric("Módel", f"{cur['model']:+.2f}%")
            for i, a in enumerate(cur["analysts"], start=1):
                cols[i].metric(a["source"], f"{a['mm_pct']:+.2f}%", help=f"y/y {a['yoy_pct']:.1f}%")
            cols[-1].metric("Samhljóða", f"{cur['consensus_mm']:+.2f}%",
                            f"{cur['model_minus_consensus']:+.2f}pp vs módel")
            st.caption("Bankagreiningar keyra sömu opinberu gögn og módelið (eldsneyti, HMS, "
                       "útsölur), svo þær eru erfiðar að slá við h=1 — forskot módelsins er á "
                       "h=3–12. / Analysts run the same public data, so they are hard to beat "
                       "at h=1; the model's edge is at h=3–12.")
        if an["track"] is not None and len(an["track"]):
            st.markdown("**Ferilskrá / track record** (m/m villa vs raun, skráðir mánuðir):")
            tr = an["track"].copy()
            tr.columns = ["Aðili", "n", "RMSE", "MAE", "Bias"]
            st.dataframe(tr.round(3), width="stretch", hide_index=True)
            if an["detail"] is not None:
                det = an["detail"].copy(); det["manudur"] = det.manudur.astype(str)
                det.columns = ["Mánuður", "Aðili", "Spá m/m", "Raun m/m", "Villa"]
                st.dataframe(det.round(3), width="stretch", hide_index=True)
            st.caption("Fáir mánuðir skráðir enn — ferilskráin styrkist eftir því sem spár "
                       "bætast við hvern mánuð.")

    # --- 6) Model vs Seðlabanki Peningamál -------------------------------
    st.header("6 · Módel vs Seðlabanki (Peningamál)")
    pm = rep.get("peningamal")
    if not pm:
        st.info("Engin Peningamál-spá skráð. Bæta við í data/benchmarks/peningamal.csv "
                "(ársfjórðungsleg y/y-spá, Tafla 5, birt ~4x á ári).")
    else:
        pmr = pd.DataFrame(pm["rows"])
        st.markdown(f"**Ársfjórðungsleg y/y-spá** — Peningamál {pd.Timestamp(pm['vintage']):%Y-%m-%d}:")
        show = pmr.copy()
        show.columns = ["Ársfj.", "Módel y/y %", "Seðlabanki y/y %", "Munur pp"]
        st.dataframe(show, width="stretch", hide_index=True)
        # chart: model monthly y/y path + Peningamál quarterly markers + target
        yy_path2 = rep["yy_path"]
        xh = head[("CPI", "change_A")]["2024":]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(xh.index.to_timestamp(), xh, color=C["black"], lw=1.6, label="raun y/y")
        ax.plot(yy_path2.index.to_timestamp(), yy_path2, color=C["blue"], lw=2.2, label="módel (reconciled)")
        qx = [pd.Period(r["quarter"], "Q").to_timestamp(how="end") for r in pm["rows"]]
        ax.plot(qx, [r["peningamal_yoy"] for r in pm["rows"]], color=C["green"],
                lw=1.6, marker="s", ms=6, ls="--", label="Seðlabanki (Peningamál)")
        ax.axhline(2.5, color=C["black"], lw=1, ls=":", alpha=0.6, label="markmið 2,5%")
        ax.set_ylabel("y/y %"); ax.set_title("Verðbólga: módel vs Seðlabanki")
        ax.legend(frameon=False, ncols=4, fontsize=8)
        st.pyplot(fig, width="stretch")
        st.caption("Nálægt í ár en módelið sér þrálátari verðbólgu en Seðlabankinn á 12 mán.: "
                   "leiguígildi (reiknuð húsaleiga) er þrálátt og heldur verðbólgu uppi, á meðan "
                   "spá Seðlabankans stefnir hraðar í markmið. / Close near-term; the model sees "
                   "stickier 12-month inflation than the bank because rental-equivalence rent is "
                   "persistent (Phase 4), while the bank's path reverts faster to target.")

    st.divider()
    st.download_button("Sækja samantekt (markdown)", _report.to_markdown(rep),
                       file_name=f"vnv_report_{last_m}.md")
