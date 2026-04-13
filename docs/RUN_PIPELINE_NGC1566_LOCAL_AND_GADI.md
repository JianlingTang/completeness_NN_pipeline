# ngc1566: clean slate, local one-liner, and Gadi PBS

This document records a **working** end-to-end command for **LEGUS `ngc1566`** (auto-download via `scripts/setup_legus_galaxy.py` when the galaxy directory is missing), plus a **Gadi (NCI)** PBS template aligned with the usual `run_pipeline.py` workflow.

---

## 1. What “delete ngc1566” means

- **Remove the galaxy tree under the project root:**  
  `rm -rf <PROJECT_ROOT>/ngc1566`
- **Remove pipeline scratch for that run (optional but recommended for a clean retry):**  
  `rm -rf <PROJECT_ROOT>/tmp_pipeline_test/ngc1566_*`  
  (temporary detection/injection dirs named by galaxy and frame.)
- **Remove old log files (optional):**  
  `rm -f <PROJECT_ROOT>/logs/ngc1566*.log`
- **Do not delete** shared `PSF_files/psf_*_wfc3_*.fits` unless you know another PSF set covers `f275w`–`f814w` for WFC3. The pipeline matches `psf_*_wfc3_<filter>.fits` (e.g. M101 PSFs for UV bands and `psf_ngc1566_wfc3_*` where present).

---

## 2a. Gadi PBS: full Phase A+B + diagnostics (reff 5, nframe 5, ncl 400)

After `.venv` exists on **login** (`pip install -e .`), submit a job whose script ends with:

```bash
export SKIP_PIP_INSTALL=1
export IRAF=/path/to/iraf
bash scripts/run_ngc1566_full_pipeline_gadi.sh
```

This runs `run_pipeline.py` with **photometry** (default), so you get completeness plots under `ngc1566/white/diagnostics/` plus optional three-panel figures per frame (`RUN_THREE_PANEL_PLOTS=1` by default). Override `OUTNAME`, `NCL`, `NFRAME`, `REFF_LIST`, `GALAXY` via `export` before the script.

---

## 2. Local (macOS/Linux): one-liner from project root

Assumes: repo root = project root, `.venv` installed, network available for STScI download on first run.

**Full pipeline:** Phase A + B (default **includes** five-filter photometry + catalogue) + **NN** (`--run_ml`). Parameters match a typical “production-like” test: `reff=5` pc, **5 frames**, **500** clusters per frame.

```bash
cd /path/to/comp_pipeline_restructure && rm -rf ngc1566 tmp_pipeline_test/ngc1566_* logs/ngc1566*.log 2>/dev/null; .venv/bin/python scripts/run_pipeline.py --galaxy ngc1566 --nframe 5 --ncl 500 --reff_list 5 --run_ml --outname run_reff5_nframe5 --parallel 2>&1 | tee logs/ngc1566_full_pipeline.log
```

Notes:

- Omitting `--skip-galaxy-setup` lets `run_pipeline.py` call `setup_legus_galaxy.py` when HLSP FITS / white / `r2_wl_aa_*.config` are missing (see `scripts/run_pipeline.py`).
- `--parallel` speeds Phase B when jobs are independent; with **default photometry**, the pipeline may still run heavy IRAF work in ways that are not fully parallel—adjust expectations on wall time.
- To **skip** photometry (detection/matching only): add `--no_photometry`.
- To **skip** auto galaxy setup (data already on disk): add `--skip-galaxy-setup`.

---

## 3. Gadi (NCI): PBS job script

Save as e.g. `run_ngc1566_pipeline.pbs`, edit **PROJECT_ROOT** and **IRAF**, then `qsub run_ngc1566_pipeline.pbs`.

This merges the usual PBS headers (project `jh2`, `expresssr`, walltime, memory, `jobfs`, storage, mail) with the same pipeline one-liner as above. Path variables follow the pattern you use for SLUG/PSF/FITS/BAO.

```bash
#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -N comp_pipeline_ngc1566
#PBS -j oe
#PBS -m bea
#PBS -l storage=scratch/jh2+gdata/jh2+scratch/mk27
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/comp_pipeline_ngc1566.o
#PBS -e /scratch/jh2/jt4478/output/comp_pipeline_ngc1566.e

set -euo pipefail

# ============ paths (edit) ============
PROJECT_ROOT="/g/data/jh2/jt4478/comp_pipeline_restructure"

# Optional overrides: point at large data outside the repo if needed
COMP_SLUG_LIB_DIR="/g/data/jh2/jt4478/SLUG_library"
COMP_PSF_PATH="/g/data/jh2/jt4478/PSF_files"
# Galaxy FITS + configs: use project root so ngc1566/ is created under the repo by setup
COMP_FITS_PATH="${PROJECT_ROOT}"
COMP_BAO_PATH="/g/data/jh2/jt4478/.deps/local/bin"
IRAF="/path/to/iraf"   # set to your IRAF root; required for default photometry

# PyRAF/IRAF for Phase B photometry
export IRAF

# Pipeline path overrides (must match login-node preflight if you use --check-only)
export COMP_SLUG_LIB_DIR
export COMP_PSF_PATH
export COMP_FITS_PATH
export COMP_BAO_PATH

# Optional: put temp on job-local storage (uncomment if your site provides PBS_JOBFS)
# export COMP_TEMP_BASE_DIR="${PBS_JOBFS:-${PROJECT_ROOT}/tmp_pipeline_test}"

cd "${PROJECT_ROOT}"

# Python (adjust modules to your Gadi stack)
# module load python3/3.11.0
source .venv/bin/activate

# Guardrail (recommended before a long job)
# python scripts/run_pipeline.py --check-only
# python scripts/run_pipeline.py --check-only   # add --no_photometry if not running photometry

python scripts/run_pipeline.py \
  --galaxy ngc1566 \
  --nframe 5 \
  --ncl 500 \
  --reff_list 5 \
  --run_ml \
  --outname run_reff5_nframe5 \
  --parallel \
  2>&1 | tee "${PROJECT_ROOT}/logs/ngc1566_full_pipeline_gadi_${PBS_JOBID:-manual}.log"
```

**First-time download on Gadi:** ensure the compute node can reach `archive.stsci.edu` (HTTP). The job creates `ngc1566/` under `COMP_FITS_PATH` (here `PROJECT_ROOT`) via `setup_legus_galaxy.py` when inputs are missing.

**Preflight:** from a login node with the **same** `export` block:

```bash
cd "${PROJECT_ROOT}" && source .venv/bin/activate
python scripts/run_pipeline.py --check-only
# With photometry checks:
python scripts/check_pipeline_paths.py --galaxy ngc1566 --run-photometry
```

---

## 4. See also

- [RUNNING_ON_HPC.md](RUNNING_ON_HPC.md) — general layout, `COMP_*`, SLURM/PBS patterns.
- [RUNNING.md](RUNNING.md) — local usage of `run_pipeline.py`.
- `scripts/package_for_hpc.sh` — bundle code for transfer to Gadi.
