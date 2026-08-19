"""Check reconstruction identities on the old-classification panel (VIS01301)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

pd.set_option("display.width", 220)
from vnv import ingest

old = ingest.load_panel_old()
head = ingest.load_headline()

codes = sorted(old.code.unique())
print("n codes:", len(codes))
print(codes)

# Identify division-level codes: IS + 2 digits
div = old[old.code.str.fullmatch(r"IS\d\d")].copy()
print("\ndivision codes:", sorted(div.code.unique()))
vsum = div.groupby("manudur").vaegi.sum()
print("\nvaegi sum stats:", vsum.min(), vsum.max())

agg = div.groupby("manudur").ahrif.sum()
hl = head[("CPI", "change_M")].reindex(agg.index)
diff = (agg - hl).dropna()
print("\nsum(ahrif) - headline m/m: max abs by year")
print(diff.abs().groupby(diff.index.year).max().to_string())

# Level reconstruction check: chain published headline m/m vs published level
idx = head[("CPI", "index")]
mm = head[("CPI", "change_M")]
recon = idx.shift(1) * (1 + mm / 100)
err = (recon - idx).dropna()
print("\nchained level from published m/m vs published level, max abs err by year (2015+):")
e = err[err.index >= "2015-01"]
print(e.abs().groupby(e.index.year).max().to_string())

# vaegi(t-1) * change identity on old panel
div = div.sort_values(["code", "manudur"])
div["vaegi_lag"] = div.groupby("code")["vaegi"].shift(1)
d = div.dropna(subset=["vaegi_lag", "manadarbreyting", "ahrif"])
print("\nold panel: ahrif vs vaegi(t-1)*chg/100 max abs err:", (d.vaegi_lag * d.manadarbreyting / 100 - d.ahrif).abs().max())
print("old panel: ahrif vs vaegi(t)*chg/100 max abs err:  ", (d.vaegi * d.manadarbreyting / 100 - d.ahrif).abs().max())
