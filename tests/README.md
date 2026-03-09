# Pipeline tests

- **Unit** (`tests/unit/`): Diagnostics (completeness bins, load match summaries), CI filter (mag1−mag3, threshold, merr), coordinate matcher, label builder. No I/O beyond temp files.
- **Integration** (`tests/integration/`): Completeness in mag/mass/age bins with synthetic logistic-like data; assertions that completeness ∈ [0,1], brighter/high-mass bins have higher completeness; optional figure save. Build ML inputs from mock pipeline outputs.
- **E2E** (`tests/e2e/`): Config import, `plot_completeness_diagnostics` with empty dir; optional full run (`RUN_PIPELINE_E2E=1`).

## Run

```bash
# All new tests (unit + integration + e2e)
pytest

# With coverage
pytest --cov=cluster_pipeline --cov-report=term-missing

# Only unit
pytest tests/unit/

# Full E2E (needs data)
RUN_PIPELINE_E2E=1 pytest tests/e2e/
```

## Completeness checks

- **Completeness in [0, 1]** (or NaN for empty bins).
- **Logistic-like**: synthetic data with detection probability logistic in mag (or log mass) → mean completeness in bright/high-mass bins ≥ faint/low-mass bins.
- **Mag / mass / age (three panels)**: The integration test `test_completeness_visual_mag_mass_age_bins` draws completeness vs **magnitude**, vs **log10(mass)**, and vs **log10(age)** and saves under pytest’s `tmp_path` (removed after the run). To get the same figure in a fixed location, run:
  ```bash
  python scripts/plot_completeness_mag_mass_age.py
  ```
  This writes **`tests/output/completeness_mag_mass_age_bins.png`** (left: mag, middle: mass, right: age). Use `--out PATH` to save elsewhere.
