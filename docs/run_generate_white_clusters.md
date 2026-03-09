# Running White-Cluster Generation and the Refactored Pipeline

After refactoring, **white synthetic image generation** and **detection / matching / dataset** are two separate phases:

- **White synthetic generation (SLUG to white light to BAOlab to FITS + coords)**  
  Lives in the **legacy script** `scripts/generate_white_clusters.py`; it has **not** been moved into `cluster_pipeline`.
- The **refactored pipeline** (`cluster_pipeline`) handles: running detection, then matching (and optional photometry / catalogue / dataset) **given existing** synthetic FITS and `white_position_*.txt`. Configuration is via **env + PipelineConfig**; entry points are **`run_galaxy_pipeline`** and **`run_ast_pipeline`**.

Below: **how to run the refactored pipeline**, **how to run the legacy script**, and **how to chain the full workflow**.

---

## 1. Refactored pipeline (cluster_pipeline): required files and how to call it

### 1.1 Entry points and required files

- **Code**: the `cluster_pipeline` package (config, matching, detection, pipeline, etc.).
- **Configuration**: via environment variables or `get_config(overrides=...)`; paths are **no longer** hardcoded (see `cluster_pipeline/config/pipeline_config.py`).
- **Inputs**: for each `(galaxy_id, frame_id, reff)`, the following must **already exist**:
  - One synthetic FITS under `config.synthetic_fits_dir(galaxy_id)` matching `*_frame{frame_id}_{outname}_reff{reff:.2f}.fits`;
  - One coords file: `config.white_dir(galaxy_id) / f"white_position_{frame_id}_{outname}_reff{reff:.2f}.txt"`.

So the refactored pipeline **does not** generate white clusters itself; it only copies these files from the existing directories into a temp dir, then runs SExtractor and matching.

### 1.2 Environment variables (and default paths)

| Variable | Meaning | Default (aligned with legacy) |
|----------|---------|------------------------------|
| `COMP_MAIN_DIR` | Main directory (parent of galaxy subdirs) | Project root |
| `COMP_FITS_PATH` | Multi-band FITS root | Same as COMP_MAIN_DIR |
| `COMP_PSF_PATH` | PSF directory | Project root / PSF_all |
| `COMP_BAO_PATH` | BAOlab root | Project root / baolab |
| `COMP_SLUG_LIB_DIR` | SLUG library directory | Project root / SLUG_library |
| `COMP_OUTPUT_LIB_DIR` | Extra SLUG library | Project root / output_lib |
| `COMP_TEMP_BASE_DIR` | Temp directory | `/tmp/cluster_pipeline` |

If unset, defaults are used; if set, the refactored pipeline uses these paths everywhere.

### 1.3 How to call (Python)

```python
from cluster_pipeline.config import get_config
from cluster_pipeline.pipeline import run_galaxy_pipeline

# Use env vars or overrides
config = get_config(overrides={"main_dir": "/your/main/directory"})

# Run one galaxy: for each (frame_id, reff) copy from synthetic_fits + white, then detection + matching
run_galaxy_pipeline(
    "ngc628-c_white-R17v100",
    config=config,
    outname="pipeline",
    run_injection=True,   # Copies existing frames/coords from config dirs
    run_detection=True,
    run_matching=True,
    keep_frames=False,
)
```

Or use the AST orchestrator (multiple frame/reff can run in parallel):

```python
from cluster_pipeline.pipeline import run_ast_pipeline

run_ast_pipeline(
    "ngc628-c_white-R17v100",
    config=config,
    outname="pipeline",
    run_injection=True,
    run_detection=True,
    run_matching=True,
    parallel=True,
)
```

**Note:** When `run_injection=True`, the pipeline only **copies** existing files from `synthetic_fits_dir` / `white_dir`; it does not call BAOlab or SLUG. Those synthetic FITS and `white_position_*.txt` must be produced beforehand (e.g. by the legacy script in the next section).

### 1.4 Running only the first N stages (max_stage)

The pipeline has 6 stages in a fixed order; use **`max_stage=N`** to run only stages 1 through N (N+1 through 6 are skipped):

| Stage | Name | Meaning |
|-------|------|---------|
| 1 | injection | Copy/generate synthetic frame + coords |
| 2 | detection | Run SExtractor on frame |
| 3 | matching | Match injected vs detected coords; write matched_coords |
| 4 | photometry | Aperture photometry (optional) |
| 5 | catalogue | CI/merr cuts; in_catalogue |
| 6 | dataset | Build final dataset parquet/npy |

Example: run only injection + detection (no matching):

```python
from cluster_pipeline.config import get_config
from cluster_pipeline.pipeline import run_galaxy_pipeline

config = get_config()
# Run only stages 1 and 2
run_galaxy_pipeline(
    "ngc628-c_white-R17v100",
    config=config,
    outname="pipeline",
    max_stage=2,
)
# Run 1 through 3 (injection + detection + matching)
run_galaxy_pipeline("ngc628-c_white-R17v100", config=config, max_stage=3)
```

If you do not pass `max_stage`, the boolean flags `run_injection`, `run_detection`, `run_matching`, etc. control what runs. `run_ast_pipeline` also supports `max_stage`.

---

## 2. Legacy script: generating white clusters (not in cluster_pipeline)

**`scripts/generate_white_clusters.py`** is still responsible for:

- Reading the SLUG library, computing white light, calling BAOlab to produce synthetic FITS;
- Writing `white_position_{frame}_{outname}_reff{reff}.txt`, etc.

Paths can be overridden via **environment variables** or **`--directory`**; see the script and `docs/RUNNING.md`. To run white-cluster generation you still need:

1. Under `--directory`: `galaxy_names.npy`, `galaxy_filter_dict.npy` (as above);
2. SLUG, FITS, PSF, and BAOlab directories present and consistent with the script (or set env / CLI paths).

**Minimal run example:**

```bash
cd /path/to/comp_pipeline_restructure

python scripts/generate_white_clusters.py \
  --gal_name ngc628 \
  --directory /path/to/your/completeness/dir \
  --outname my_run
```

For more options (`--galaxy_fullname`, `--ncl`, `--mrmodel`, `--eradius_list`, `--validation`, etc.) see the script's `argparse` help or the run instructions in this repo.

---

## 3. Full workflow: legacy white generation then refactored pipeline

1. **Generate synthetic FITS + white coords with the legacy script**  
   Run **scripts/generate_white_clusters.py** so that outputs go to the same place as `COMP_MAIN_DIR` / `COMP_FITS_PATH` (or copy outputs into the `main_dir` / `galaxy_id/white/` etc. used by config).
2. **Set environment variables** (or `get_config(overrides=...)`) so that `synthetic_fits_dir` and `white_dir` point to the newly generated files.
3. **Run the refactored pipeline** for detection + matching:
   - `run_galaxy_pipeline(galaxy_id, config=config, run_injection=True, run_detection=True, run_matching=True)`, or  
   - `run_ast_pipeline(...)`.

That is the full workflow: run legacy white-cluster generation once, then run the refactored pipeline for the rest.

---

## 4. Running only stages 1–3 and plotting diagnostics (completeness vs magnitude)

To run only **stages 1, 2, 3** (injection, detection, matching) and then plot **x = magnitude, y = completeness**, do the following.

### Option A: Command-line script (recommended)

```bash
cd /path/to/comp_pipeline_restructure

# Run stages 1–3 then plot and save
python scripts/run_stage123_and_plot_diagnostics.py \
  --galaxy ngc628-c_white-R17v100 \
  --outname pipeline \
  --save completeness_diagnostics.png
```

If you have already run stages 1–3 and only want to re-plot:

```bash
python scripts/run_stage123_and_plot_diagnostics.py \
  --galaxy ngc628-c_white-R17v100 \
  --outname pipeline \
  --no-pipeline \
  --save completeness_diagnostics.png
```

### Option B: Call from Python

```python
from cluster_pipeline.config import get_config
from cluster_pipeline.pipeline import run_galaxy_pipeline, plot_completeness_diagnostics
import matplotlib.pyplot as plt

config = get_config()

# 1) Run only stages 1–3
run_galaxy_pipeline(
    "ngc628-c_white-R17v100",
    config=config,
    outname="pipeline",
    max_stage=3,
    keep_frames=False,
)

# 2) Plot diagnostics: x = magnitude, y = completeness
ax = plot_completeness_diagnostics(
    "ngc628-c_white-R17v100",
    config,
    outname="pipeline",
    n_bins=15,
)
ax.get_figure().savefig("completeness_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Notes

- **Magnitude source**: from the third column of the injection coords file (e.g. `white_position_*.txt`); the legacy script writes (y x mag).
- **Completeness**: per magnitude bin, completeness = number of matched injected sources / total injected in that bin.
- **Diagnostic table location**: `{main_dir}/{galaxy_id}/white/diagnostics/match_summary_frame*_reff*_{outname}.txt`.

---

## 5. One-command setup for all dependencies

Python packages and external tools can be set up via:

```bash
# Full setup (recommended): venv + pip + SExtractor/BAOlab checks
make setup
# or:
bash scripts/setup_env.sh

# Quick setup (Python packages only, skip external tools)
make setup-quick
# or:
bash scripts/setup_env.sh --quick
```

The script will:

1. Create a `.venv` and install all packages from `requirements.txt`
2. **Install SExtractor**: try brew (macOS), then apt (Ubuntu), then build from GitHub source (`--disable-model-fitting`, no ATLAS/FFTW)
3. **Install BAOlab**: clone from GitHub (soerenslarsen/baolab), unpack tarball, `make`, install to `.deps/local/bin/bl`

**Removed dependencies:**

- **slugpy**: no longer required. SLUG library FITS reading is in `cluster_pipeline.data.slug_reader` (pure Python, astropy + embedded HST filter Vega zeropoint table). No GSL or C build needed.
- **pyraf**: not required for the pipeline except for stage 4 (aperture photometry via IRAF daophot). **scripts/generate_white_clusters.py** uses pure astropy for FITS arithmetic (`cluster_pipeline.utils.fits_arithmetic`).

### Makefile shortcuts

| Command | Purpose |
|---------|---------|
| `make setup` | Full setup |
| `make setup-quick` | Python packages only |
| `make test` | Run pytest |
| `make lint` | flake8 + mypy |
| `make ci` | lint + test |
| `make clean` | Clean venv and caches |

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Refactored? | Yes. Detection, matching, config, manifest, dataset, etc. are in `cluster_pipeline`, config-driven, no hardcoded paths. |
| Where is "generate white clusters"? | In **scripts/generate_white_clusters.py**; not yet moved into `cluster_pipeline`. |
| What files and how to call after refactor? | You need **existing** synthetic FITS and `white_position_*.txt`; set **env or get_config(overrides)**; call **run_galaxy_pipeline** or **run_ast_pipeline** from Python. |
| How to actually run "generate white" once? | Run **python scripts/generate_white_clusters.py**; have the npy files and SLUG/FITS/PSF/BAOlab paths under your directory (or set env/CLI paths). |
| How to install all dependencies? | `make setup` or `bash scripts/setup_env.sh`. No need to install slugpy, GSL, or pyraf for the pipeline (pyraf only for stage 4 photometry). |
