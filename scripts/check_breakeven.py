import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnv import benchmarks, lanamal, report

print("=== breakeven curve (today) ===")
cur = lanamal.breakeven_curve(use_cache=False)
print(cur.to_string(index=False))

print("\nwriting slot:", lanamal.update_breakeven_slot(use_cache=True))
print("\nloaded back:")
print(benchmarks.load_breakeven().to_string(index=False))

rep = report.build_report()
print("\nmodel 12m inflation:", round(rep["yy_12m"], 2), "%")
mvb = benchmarks.model_vs_breakeven(rep["yy_12m"])
print("model_vs_breakeven:", mvb)
