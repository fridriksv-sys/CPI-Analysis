# VNV model — agent context

Bottom-up Icelandic CPI (VNV) forecast. Full plan in `PLAN.md`; read it before
structural changes. Hard rules:

- **Phase 2 gate**: any change to ingestion/reconstruction must keep
  `notebooks/02_reconstruction_check.ipynb` all-PASS. No forecasting code on a
  failing gate.
- **Never hardcode a PxWeb table ID or weight you have not fetched.** Discovery
  via `px_client.discover_tables()`; discovered IDs logged to `config/px_tables.yaml`.
- **No synthetic/placeholder data**, including tests. If a fetch fails, stop and
  report; never substitute plausible numbers.
- Icelandic identifiers where they match the source (vaegi, ahrif, manadarbreyting);
  comments in English.
- Weights: published monthly Vægi % is price-updated (value share). Contribution
  identity m/m = Σ vægi(t−1)·change(t); at base changes (April ≤2024, Jan ≥2025)
  de-update vægi(t) by the component's own change — one formula, all months.
- Reiknuð húsaleiga (CP042): June-2024 methodology break — never fit across it
  without a regime split.
- API precision ceiling: 1dp indices, 2dp weights/changes. All tolerances must be
  rounding-aware; a "perfect" backtest (<0.08pp h=1 RMSE) means look-ahead leakage.
- Run venv python: `.venv\Scripts\python.exe`. Rebuild notebooks:
  `.venv\Scripts\python.exe scripts\build_notebooks.py [1|2|3]`.
