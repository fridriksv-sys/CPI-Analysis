"""Quick exploration: fetch everything, check precision, weight conventions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
from vnv import ingest

head = ingest.load_headline()
sub = ingest.load_sub_new()

DIV_RE = r"CP(0[1-9]|1[0-3])"  # divisions CP01..CP13, excluding CP00 (= headline)

last = sub[sub.manudur == sub.manudur.max()]
print("=== latest month:", sub.manudur.max(), "===")
print(last[last.code.str.fullmatch(DIV_RE)][["code", "heiti", "visitala", "vaegi", "manadarbreyting", "ahrif"]].to_string())

m = sub[sub.code.str.fullmatch(DIV_RE)].copy().sort_values(["code", "manudur"])
m["vaegi_lag"] = m.groupby("code")["vaegi"].shift(1)
mm = m.dropna(subset=["vaegi_lag", "manadarbreyting", "ahrif"])
print("\n=== ahrif convention (max abs err across all divisions/months) ===")
print("ahrif vs vaegi(t)*chg/100:   ", (mm.vaegi * mm.manadarbreyting / 100 - mm.ahrif).abs().max())
print("ahrif vs vaegi(t-1)*chg/100: ", (mm.vaegi_lag * mm.manadarbreyting / 100 - mm.ahrif).abs().max())

agg = m.groupby("manudur").ahrif.sum()
hl = head[("CPI", "change_M")].reindex(agg.index)
cp00 = sub[sub.code == "CP00"].set_index("manudur")
print("\n=== sum(division ahrif) vs published headline m/m ===")
print(pd.DataFrame({"sum_ahrif": agg, "headline_mm": hl, "diff": agg - hl}).to_string())

print("\n=== precision ===")
print("headline index sample:", head[("CPI", "index")].dropna().tail(3).tolist())
print("sub visitala sample:  ", last.visitala.dropna().head(6).tolist())
print("vaegi sample:         ", last.vaegi.dropna().head(6).tolist())
print("CP00 visitala tail:   ", cp00.visitala.tail(3).tolist())
print("division vaegi sum by month:")
print(m.groupby("manudur").vaegi.sum().to_string())
