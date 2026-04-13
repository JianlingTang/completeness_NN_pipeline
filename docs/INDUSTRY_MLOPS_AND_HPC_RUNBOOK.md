# Industry MLOps Mapping and HPC One-Click Runbook

This document has two goals:

1. Save a practical "industry tools comparison" for this project.
2. Provide a copy-paste runbook to package locally and run full pipeline on HPC
   (from `run_pipeline.py` through NN training artifacts).

---

## 1) Industry Tooling Comparison (without changing scientific core)

Core kept unchanged:
- `scripts/run_pipeline.py`
- `scripts/build_ml_inputs.py`
- `scripts/perform_ml_to_learn_completeness.py`
- current science configs/thresholds/formulas

| Capability | Current project | Industry tools | Fit for this repo | Benefit | Boundary / caveat |
|---|---|---|---|---|---|
| Orchestration | PBS shell chaining | Airflow / Prefect / Dagster / Step Functions | High | Retries, dependencies, alerts | Extra control-plane setup |
| Experiment tracking | logs + files | MLflow / Weights & Biases | Very high | Param/metric/artifact lineage | Need naming/tag conventions |
| Data+artifact versioning | folder naming | DVC / lakeFS | High | Reproducible input-output snapshots | Storage layout discipline needed |
| Batch compute abstraction | direct PBS | AWS Batch / K8s Jobs | Medium | Elastic autoscaling | Migration effort for I/O paths |
| Containerized runtime | venv + modules | Docker + Apptainer | High on HPC | Stable runtime across nodes | IRAF/legacy tools may need special image handling |
| Model registry | manual files | MLflow Registry / SageMaker Registry | High | Stage/prod promotion workflow | Governance process overhead |
| CI/CD | manual run | GitHub Actions / GitLab CI | High | Automated test/build/release | Need tests + release policy |
| Infra as code | manual config | Terraform / CDK / Pulumi | Medium-high | Repeatable infra and permissions | IaC learning curve |
| Monitoring | stdout/stderr | CloudWatch / Datadog / Prometheus | Medium-high | Runtime visibility, alerting | Instrumentation effort |

Recommended first 5 upgrades:
1. MLflow (or W&B) tracking
2. GitHub Actions CI for test/build
3. DVC for key input/output lineage
4. Apptainer image for HPC reproducibility
5. DAG orchestrator for multi-stage dependencies

---

## 2) HPC Packaging and One-Click Full Run

### 2.1 Package locally

From repo root:

```bash
bash scripts/package_for_hpc.sh --mode minimal
```

Output example:
- `hpc_bundle_minimal_YYYYMMDD_HHMMSS.tar.gz`

Use `--mode full` only if you really want to ship large datasets inside the tarball.

### 2.2 Upload and unpack on HPC

```bash
scp hpc_bundle_minimal_*.tar.gz <user>@<hpc>:/path/to/workdir/
ssh <user>@<hpc>
cd /path/to/workdir
tar -xzf hpc_bundle_minimal_*.tar.gz
cd comp_pipeline_restructure
```

### 2.3 Prepare environment on HPC

```bash
module load python3/3.11.0
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set required paths (edit to your site):

```bash
export COMP_SLUG_LIB_DIR=/g/data/<proj>/SLUG_library
export COMP_PSF_PATH=/g/data/<proj>/PSF_files
export COMP_FITS_PATH=/g/data/<proj>/ngc_data
export COMP_BAO_PATH=/g/data/<proj>/.deps/local/bin
export IRAF=/path/to/iraf   # needed when running photometry
```

Preflight check:

```bash
python scripts/check_pipeline_paths.py --run-photometry
```

### 2.4 Initialize one LEGUS galaxy directory (pure Python, no IDL)

Use this to create `<project_root>/<galaxy>/`, download LEGUS FITS + deterministic readme,
extract archives, and build `<galaxy>_white.fits` with Python:

```bash
python scripts/setup_legus_galaxy.py \
  --galaxy ngc1313-e \
  --project-root /g/data/jh2/jt4478/comp_pipeline_restructure
```

The setup script also syncs key metadata files into galaxy root (for pipeline compatibility):
- `automatic_catalog_<gal>.readme`
- `avg_aperture_correction_<gal>.txt`
- `header_info_<gal>.txt`
- `r2_wl_aa_<gal>.config`

If auto-discovery misses a page link, pass direct URLs:

```bash
python scripts/setup_legus_galaxy.py \
  --galaxy ngc1313-e \
  --project-root /g/data/jh2/jt4478/comp_pipeline_restructure \
  --fits-url "https://archive.stsci.edu/.../file1.tar.gz" \
  --fits-url "https://archive.stsci.edu/.../file2.fits.gz"
```

---

## 3) One-click dependency chain: CPU -> GPU -> (optional) PyPI

Scripts already provided:
- CPU pipeline job: `scripts/run_pipeline_pbs.sh`
- GPU NN job: `scripts/run_nn_gpu_pbs.sh`
- Submit CPU->GPU: `scripts/submit_cpu_then_gpu.sh`
- Publish PyPI job: `scripts/run_publish_pypi_pbs.sh`
- Submit CPU->GPU->PyPI: `scripts/submit_cpu_gpu_pypi.sh`

### 3.1 Edit job scripts once

At minimum, update:
- `PROJECT_ROOT`
- `VENV` (if used)
- queue/resource fields (`#PBS -q`, `ncpus`, `ngpus`, `walltime`, ...)
- run parameters (`NFRAME`, `REFF_LIST`, `NCL`, `sigma_pc`)

For your target:
- `NFRAME=5`
- `REFF_LIST="1,2,3,4,5,6,7,8,9,10"`
- `NCL=500`

### 3.2 Submit CPU->GPU

```bash
bash scripts/submit_cpu_then_gpu.sh
```

This uses:
- `qsub scripts/run_pipeline_pbs.sh`
- `qsub -W depend=afterok:<cpu_job_id> scripts/run_nn_gpu_pbs.sh`

GPU starts only if CPU job exits with code 0.

### 3.3 Submit CPU->GPU->PyPI (optional)

```bash
export PYPI_API_TOKEN='pypi-xxxxx'
bash scripts/submit_cpu_gpu_pypi.sh
```

PyPI job will run only after GPU success.

---

## 4) Where outputs are saved

### 4.1 Pipeline outputs (including parquet)

- Catalogue parquet:
  - `ngc628-c/white/catalogue/catalogue_frame*_test_reff*.parquet`
  - `ngc628-c/white/catalogue/match_results_frame*_test_reff*.parquet`
  - `ngc628-c/white/catalogue/photometry_frame*_test_reff*.parquet`
- Detection labels:
  - `ngc628-c/white/detection_labels/*.npy`
- Diagnostics:
  - `ngc628-c/white/diagnostics/*`

### 4.2 ML inputs

- `det_3d_<galaxy>.npy` (or configured path)
- `allprop_<galaxy>.npz` (or configured path)

### 4.3 Trained NN artifacts (weights/biases included in checkpoints)

In `ML_OUT_DIR` from `scripts/run_nn_gpu_pbs.sh`, typically:
- `best_model_phys_<outname>.pt`
- `best_model_phot_<outname>.pt`
- `scaler_phys_<outname>.pkl`
- `scaler_phot_<outname>.pkl`
- sweep plots and grids (`*.png`, `*.npz`)

Notes:
- `.pt` checkpoint files contain learned network parameters (weights and biases).
- This repo does not currently log to Weights & Biases (wandb) service by default.

---

## 5) Quick sanity checklist before big runs

1. `python scripts/check_pipeline_paths.py --run-photometry` passes.
2. Single small dry run succeeds (`nframe=1`, one `reff`).
3. Queue resources fit your project quota.
4. Output dirs are under expected storage mount.
5. If publishing, bump `pyproject.toml` version before PyPI upload.

