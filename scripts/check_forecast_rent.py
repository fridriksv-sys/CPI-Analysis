"""Headline forecast with vs without the Phase 4 rent model routing CP042."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

pd.set_option("display.width", 200)
from vnv import ingest, models, rent

spl = ingest.load_sub_spliced()
new = ingest.load_sub_new()
old = ingest.load_panel_old()
head = ingest.load_headline()

g = models.build_component_history(spl, old, new)
last_m = g.index.max()
w0 = new[(new.manudur == last_m) & new.code.isin(models.COMPONENTS)].set_index("code").vaegi
fits = {c: models.fit_component(g[c], c) for c in models.COMPONENTS}
hms_mm = rent.hms_rent_mm()

fc_generic = models.forecast_components(fits, last_m, 12)
fc_rent = models.forecast_components(fits, last_m, 12, hms_rent_mm=hms_mm, sub_spliced=spl)

for label, fc in [("generic CP042", fc_generic), ("HMS rent model", fc_rent)]:
    hm, _, _ = models.aggregate_bottom_up(fc, w0)
    infl = ((1 + hm / 100).prod() - 1) * 100
    print(f"{label:16s}: CP042 h1={fc.loc[fc.index[0], 'CP042']:+.3f}  "
          f"12m headline inflation={infl:.2f}%")

print("\nCP042 path comparison (m/m %):")
cmp = pd.DataFrame({"generic": fc_generic["CP042"], "hms_model": fc_rent["CP042"]})
print(cmp.round(3).to_string())
