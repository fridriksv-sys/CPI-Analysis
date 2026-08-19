import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnv import ingest

new = ingest.load_sub_new()
anchors = {"CP01": 15.0, "CP03": 3.6, "CP05": 4.3, "CP042": 21.7, "CP07332": 2.55,
           "CP07222": 1.7, "CP07221": 0.8, "CP112": 0.6}
for c, a in anchors.items():
    s = new[new.code == c].set_index("manudur").vaegi
    ok = (s.min() - 0.2) <= a <= (s.max() + 0.2)
    print(f"{c:9s} anchor={a:5.2f}  range 2025-26: [{s.min():5.2f}, {s.max():5.2f}]  {'OK' if ok else 'OUT'}")
