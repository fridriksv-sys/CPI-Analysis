"""Phase 2 gate: run both reconstructions and print acceptance results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

pd.set_option("display.width", 220)
from vnv import ingest, reconstruct

old = ingest.load_panel_old()
new = ingest.load_sub_new()
wnew = ingest.load_weights_new()
head = ingest.load_headline()

# --- Era 1: old classification, 2003-2025 ---
r1 = reconstruct.mm_from_panel(old, reconstruct.DIV_OLD)
r1 = r1[r1.index >= "2019-01"]
print("=== OLD ERA (2019-2025): reconstructed m/m vs published ===")
print("months:", len(r1))
print("max |error| pp:", r1.error.abs().max().round(4), " mean error:", r1.error.mean().round(5))
print("by year:")
print(r1.error.abs().groupby(r1.index.year).max().round(4).to_string())
print("worst months:")
print(r1.reindex(r1.error.abs().nlargest(5).index)[["recon_mm", "published_mm", "error"]])

budget = reconstruct.rounding_budget_mm(pd.Series(100.0 / 12, index=reconstruct.DIV_OLD), 12)
print("rounding budget (worst case, pp):", round(budget, 4))

# level chain 2019-2025 vs published, re-anchored each December
idx = head[("CPI", "index")]
lvl = reconstruct.levels_from_mm(r1.recon_mm, pd.Period("2018-12", "M"), idx[pd.Period("2018-12", "M")])
lvl_err = (lvl - idx.reindex(lvl.index)).dropna()
print("\nlevel chain from Dec-2018 anchor: max |err| index points by year:")
print(lvl_err.abs().groupby(lvl_err.index.year).max().round(3).to_string())

# --- Era 2: COICOP2018 fixed-base, 2026- ---
w_dec25 = wnew[wnew.code.isin(reconstruct.DIV_NEW)].set_index("code")["2025M12"]
print("\n=== NEW ERA (2026-): fixed-base Laspeyres levels ===")
print("Dec-2025 division weights sum:", w_dec25.sum())
r2 = reconstruct.levels_new_era(new, w_dec25)
print(r2.round(3).to_string())
print("max |error| index points:", r2.error.abs().max().round(4))

# m/m identity on new panel too
r3 = reconstruct.mm_from_panel(new, reconstruct.DIV_NEW)
r3 = r3[r3.index >= "2026-01"]
print("\nnew era m/m identity: max |error| pp:", r3.error.abs().max().round(4))

# --- VNV an husnaedis check (old era): IS04 excluded except IS045 ---
# Published CPILH m/m from VIS01000
codes_lh = [c for c in reconstruct.DIV_OLD if c != "IS04"] + ["IS045"]
div = old[old.code.isin(codes_lh)].copy().sort_values(["code", "manudur"])
div["w_pre_raw"] = div["vaegi"] / (1 + div["manadarbreyting"] / 100)
g = div.groupby("manudur").apply(
    lambda d: (d.w_pre_raw / d.w_pre_raw.sum() * d.manadarbreyting).sum(), include_groups=False
).to_frame("recon_mm")
n = div.groupby("manudur").size()
g = g[n.reindex(g.index) == len(codes_lh)]
pub_lh = head[("CPILH", "change_M")].reindex(g.index)
g["published_mm"] = pub_lh
g["error"] = g.recon_mm - g.published_mm
g = g.dropna(subset=["error"])
g = g[g.index >= "2019-01"]
print("\n=== VNV an husnaedis (2019-2025): reconstructed vs published m/m ===")
print("max |error| pp:", g.error.abs().max().round(4), " mean:", g.error.mean().round(5))
print(g.error.abs().groupby(g.index.year).max().round(4).to_string())
