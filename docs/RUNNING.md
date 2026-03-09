# Running the Cluster Completeness Pipeline

This document describes how to run the pipeline, what files and directories are required, and the recommended order of steps.

---

## Prerequisites

- **Python** 3.10+
- **External tools** (install separately; paths are configurable):
  - **SExtractor** – source detection on synthetic FITS
  - **IRAF / PyRAF** – aperture photometry (optional, for 5-filter + CI cut)
  - **BAOlab** – synthetic source injection (e.g. under `.deps/local/bin` or set `COMP_BAO_PATH`)

---

## Required Files and Directories

Place these in your project root (or set environment variables; see below).

### Config / metadata (required)

| File or directory       | Description |
|-------------------------|-------------|
| `galaxy_filter_dict.npy` | Pickled dict: galaxy id → (list of filter names, list of instrument names). Used for filter order and SLUG/PSF lookup. |
| `galaxy_names.npy`       | Optional; used when processing multiple galaxies or auto-check mode. |
| `ngc628-c/ngc628-c_white.fits` | White-light science frame for injection (or pass `--sciframe PATH`). |
| `ngc628-c/r2_wl_aa_ngc628-c.config` | SExtractor config for white-light run. |
| `output.param`           | SExtractor output parameter file. |
| `default.nnw`            | SExtractor neural-network weight file. |
| `ngc628-c/automatic_catalog*_ngc628.readme` | Readme with aperture radius, distance modulus, CI; used by photometry and inject script. |
| `ngc628-c/header_info_ngc628.txt` | Zeropoint / exposure info per filter (optional, for some paths). |

### Data (required for injection and photometry)

| File or directory | Description |
|-------------------|-------------|
| `SLUG_library/`   | SLUG cluster library. Must contain at least `flat_in_logm` (or `flat_in_logm_cluster_phot.fits`). For non-flat mass–radius models, additional `*_cluster_phot.fits` may be needed. |
| `PSF_files/` or `PSF_all/` | Directory of PSF FITS files (e.g. `psf_*_wfc3_*.fits`). Used by BAOlab and injection. |
| `.deps/local/bin/bl` (or BAOlab path) | BAOlab executable for synthetic image generation. |

### Optional (for 5-filter photometry)

| File or directory | Description |
|-------------------|-------------|
| `ngc628-c/<filter>/` | Per-filter science FITS (e.g. `*F275W*drc.fits`) and synthetic_fits output dir. Required only when using `--run_photometry`. |

Paths can be overridden by **environment variables** (no hardcoded absolute paths):

- `COMP_MAIN_DIR` – main run directory (default: project root)
- `COMP_FITS_PATH` – root for LEGUS FITS images
- `COMP_PSF_PATH` – PSF directory
- `COMP_BAO_PATH` – BAOlab binary directory
- `COMP_SLUG_LIB_DIR` – SLUG library directory
- `COMP_OUTPUT_LIB_DIR` – additional SLUG output library (for non-flat models)

---

## How to Run

All commands assume the project root is the current working directory and the Python environment is activated.

### 1. Full pipeline (injection + detection + matching)

```bash
python scripts/run_small_test.py --cleanup --nframe 2 --reff_list "1,3,6,10"
```

- `--cleanup`: Remove previous pipeline outputs before running.
- `--nframe`: Number of frames to simulate.
- `--reff_list`: Comma-separated effective radii in pc (e.g. `"1,3,6,10"`).

This runs **Phase A** (white-light injection via `generate_white_clusters.py`) and **Phase B** (detection with SExtractor, coordinate matching). Outputs go to `ngc628-c/white/` (synthetic_fits, matched_coords, detection_labels, etc.) and `physprop/`.

### 2. With 5-filter photometry and CI cut

```bash
python scripts/run_small_test.py --cleanup --nframe 2 --reff_list "1,3,6,10" --run_photometry
```

Requires per-filter science FITS and BAOlab. Matched clusters are injected onto 5-filter images, then aperture photometry and CI cut are run; final detection labels and catalogue parquet are written.

### 3. Using pre-defined coordinates (no SLUG sampling)

```bash
python scripts/run_small_test.py --input_coords path/to/coords.txt --nframe 1 --reff_list "3"
```

`coords.txt` format: one line per cluster, `x y mag` (optionally `x y mag mass age` for 5 columns). Clusters are injected at these positions with the given white mag.

### 4. Build ML inputs from pipeline outputs

From the detection labels and physprop `.npy` files, build the 3D detection array and property `.npz` for the NN:

```bash
python scripts/build_ml_inputs.py --main-dir . --galaxy ngc628-c --outname test \
  --nframe 2 --reff-list 1 3 6 10 \
  --out-det det_3d.npy --out-npz allprop.npz
```

Use `--use-white-match` to build labels from white-match detection (detection rate) instead of post–CI labels. The script prints the suggested `perform_ml_to_learn_completeness.py` command.

### 5. Train the completeness NN

```bash
python perform_ml_to_learn_completeness.py \
  --det-path det_3d.npy \
  --npz-path allprop.npz \
  --out-dir ./nn_sweep_out \
  --clusters-per-frame 500 \
  --nframes 2 \
  --nreff 4 \
  --prop-flatten-order CFR \
  --save-best
```

Outputs: best model, scalers, and diagnostic plots under `--out-dir`. No IRAF/BAOlab required for this step.

---

## Standalone scripts (optional)

- **`extract_white.py`** – Legacy white extraction / batch processing. Use `--directory` or `COMP_MAIN_DIR` for the run directory.
- **`perform_photometry_ci_cut_on_5filters.py`** – Standalone photometry + CI reference; same logic as the pipeline’s photometry step.
- **`scripts/plot_completeness_mag_mass_age.py`** – Plot completeness vs magnitude, mass, and age (synthetic demo).
- **`scripts/sample_slug_white_mag.py`** – Sample SLUG clusters and compute white-light mag for BAOlab/`--input_coords`.

---

## Tests and lint

```bash
# Lint
ruff check .

# Tests (unit + integration + e2e)
pytest
```

By default, pytest runs `tests/unit`, `tests/integration`, and `tests/e2e` (see `pyproject.toml` or `pytest.ini`).
