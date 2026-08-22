"""Phase 7: horizon backtest, benchmarks, block error decomposition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
from vnv import backtest, ingest

head = ingest.load_headline()
print("running expanding-window backtest (this refits every jump-off)...")
bt = backtest.run_backtest(start="2025-01", max_h=12)
print(f"forecast rows: {len(bt)}  jump-offs: {bt.jump_off.nunique()}")

print("\n=== RMSE by horizon (m/m, pp) — model vs seasonal-naive floor ===")
rbh = backtest.rmse_by_horizon(bt)
print(rbh[["n", "RMSE_model", "RMSE_seasonal_naive", "MAE_model", "MAE_seasonal_naive"]].round(3).to_string())

print("\n=== h=12 y/y: model vs RW-on-y/y ===")
yb = backtest.yoy_backtest(bt, head, max_h=12)
if len(yb):
    print(f"n={len(yb)}  model RMSE={np.sqrt(((yb.model_yoy-yb.actual_yoy)**2).mean()):.3f}  "
          f"rw RMSE={np.sqrt(((yb.rw_yoy-yb.actual_yoy)**2).mean()):.3f}")
    yb2 = yb.copy(); yb2["jump_off"] = yb2.jump_off.astype(str)
    print(yb2.round(2).to_string())
else:
    print("(insufficient h=12 realized months in the COICOP2018 sample yet)")

print("\n=== h=1 error decomposition by driver block (which block drives misses) ===")
be = backtest.h1_block_error()
print(be.round(4).to_string())
