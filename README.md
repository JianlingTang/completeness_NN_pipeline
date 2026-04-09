# Cluster Completeness Pipeline

Pipeline for synthetic cluster injection, detection, matching, 5-filter photometry with CI cut, and neural-network completeness learning. Used to measure and model detection completeness as a function of magnitude, mass, and age.

## Overview

1. **Pipeline (stages 1–5):** Inject synthetic clusters on white-light and 5-filter images → run SExtractor → match injected vs detected positions → run IRAF aperture photometry → apply concentration-index (CI) cut → write detection labels and catalogue.
2. **Build ML inputs:** Assemble 3D detection array and property `.npz` from pipeline outputs (cluster-frame-radius order).
3. **NN training:** Train an MLP to predict completeness from physical and photometric features; save best model and diagnostics.

## Requirements

- **Python** 3.10+
- **External binaries:** SExtractor, IRAF/PyRAF (for aperture photometry), BAOlab (for injection). Install separately; paths are configurable (e.g. BAOlab under `.deps/local/bin`).
- **Data (not in repo):** Galaxy FITS, `galaxy_filter_dict.npy`, readme with zeropoints/CI, SLUG library, PSF files. See **`docs/RUNNING.md`** for the complete required-files list.

## Installation

```bash
git clone <your-repo-url>
cd cluster-completeness-pipeline
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[api]"
```

For the full pipeline you also need IRAF/PyRAF, SExtractor, and BAOlab; see your institution’s setup or `docs/DEPLOY_FOR_PAPER.md`.

## Using pre-trained neural networks to predict completeness using own inputs
The package is available on PyPI and can be installed via 
```bash
pip install cluster-completeness-pipeline 
```
Predictions can then be obtained using 
```python
from completeness_nn_api import ngc628_completeness_predict as predict
predict(phys=(log(mass), log(age), av)) or predict(phot=magnitudes)
```
A detailed description of the inputs, outputs, and function arguments is provided in the package documentation. 

## Quick start

See **`docs/RUNNING.md`** for the full list of required files and step-by-step run instructions.

### 1. Run the pipeline

Entry point: `scripts/run_pipeline.py`. It runs cleanup (optional), Phase A (white injection), Phase B (detection, matching, optional 5-filter inject + photometry + catalogue), and optional completeness plots.

```bash
python scripts/run_pipeline.py --cleanup --nframe 2 --reff_list "1,3,6,10" --run_photometry
```

- `--cleanup`: Remove previous pipeline outputs before running.
- `--nframe`: Number of frames.
- `--reff_list`: Comma-separated effective radii (e.g. `"1,3,6,10"`).
- `--run_photometry`: Run 5-filter injection, photometry, and CI cut (otherwise only detection + matching).

### 2. Build ML inputs

From pipeline outputs, build `det_3d.npy` and `allprop.npz` for the NN:

```bash
python scripts/build_ml_inputs.py --main-dir . --galaxy ngc628-c --outname test \
  --nframe 2 --reff-list 1 3 6 10 \
  --out-det det_3d.npy --out-npz allprop.npz
```

Use `--use-white-match` to use white-match detection labels (detection rate) instead of post–CI labels. The script prints the exact `scripts/perform_ml_to_learn_completeness.py` command to run next.

### 3. Train the NN

```bash
python scripts/perform_ml_to_learn_completeness.py \
  --det-path det_3d.npy \
  --npz-path allprop.npz \
  --out-dir ./nn_sweep_out \
  --clusters-per-frame 500 \
  --nframes 2 \
  --nreff 4 \
  --prop-flatten-order CFR \
  --save-best
```

Outputs: best model, scalers, and plots under `--out-dir`. Dependencies: `torch`, `numpy`, `scikit-learn`, `joblib`, `matplotlib` (no IRAF/BAOlab needed for this step).

## Repository layout (code only)

| Path | Description |
|------|-------------|
| **`scripts/`** | All runnable scripts; see **`scripts/README.md`** and **`docs/SCRIPTS.md`**. Entry points: `run_pipeline`, `deploy-completeness`, `serve-completeness-api`, and scripts: `run_pipeline.py`, `deploy.py`, `generate_white_clusters.py`, … |
| **`completeness_nn_api/`** | Completeness NN API: `from completeness_nn_api import ngc628_completeness_predict`, HTTP server, deploy. |
| **`checkpoints/`** | Optional: put the four NN checkpoint files here and run `deploy-completeness` (see `checkpoints/README.md`). |
| **`cluster_pipeline/`** | Config, data loaders, detection, matching, pipeline, photometry, catalogue, utils |
| **`docs/`** | RUNNING, PIPELINE_FILES, SCRIPTS (script index + inputs/outputs), FILES_FOR_GIT, DEPLOY, ARCHITECTURE, INSTALL_IRAF, COMPLETENESS_FIGURE |
| **`tests/`** | Unit, integration, and E2E tests |

Data (FITS, SLUG library, PSF, etc.) and large outputs are not in the repo; see `docs/DEPLOY_FOR_PAPER.md` and `docs/FILES_FOR_GIT.md`. **Input/output files per stage:** `docs/PIPELINE_FILES.md`.

## Tests and lint

```bash
# Lint
ruff check .

# Tests (unit + integration + e2e smoke)
pytest
```

CI runs on push/PR: `ruff check` and `pytest` (see `.github/workflows/ci.yml`).

## Documentation

- **`docs/RUNNING.md`** – **How to run the pipeline**: required files and directories, step-by-step run commands, environment variables.
- **`docs/PIPELINE_FILES.md`** – Input/output files per pipeline stage.
- **`docs/SCRIPTS.md`** – Script index with inputs/outputs and locations.
- **`docs/DEPLOY_FOR_PAPER.md`** – What to include for paper/GitHub: pipeline modules, ML step, optional reference script, exclude list.
- **`docs/FILES_FOR_GIT.md`** – Explicit list of files to commit for a pipeline-only push.
- **`docs/PUBLISH.md`** – Publish the package to PyPI and host the API (Docker, Railway, Render, etc.).
- **`docs/ARCHITECTURE.md`** – Pipeline architecture and refactor design.
- **`docs/INSTALL_IRAF.md`** – Local IRAF install for 5-filter photometry.
- **`docs/COMPLETENESS_FIGURE.md`** – Completeness workflow, scripts, and assumptions.
- **`scripts/README.md`** – Script quick reference and pointers to docs.
- **`tests/README.md`** – How to run tests and the completeness visualisation script.

## License

See repository or paper for license terms.
