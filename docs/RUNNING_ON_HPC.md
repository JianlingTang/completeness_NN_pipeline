# Running the pipeline on HPC

This guide assumes a typical HPC cluster with a batch scheduler (SLURM or PBS), shared filesystem, and optional modules for Python/SExtractor. Adapt paths and module names to your site.

---

## 0. One-command packaging (local -> HPC)

From your local repo root, create a transfer tarball:

```bash
bash scripts/package_for_hpc.sh --mode minimal
```

This creates `hpc_bundle_minimal_*.tar.gz` in the repo root.

- `--mode minimal` (recommended): code + docs + configs, excludes heavy data/outputs
- `--mode full`: includes project data for immediate run (large archive)
- `--output /abs/path/name.tar.gz`: custom output path

On HPC:

```bash
tar -xzf hpc_bundle_minimal_*.tar.gz
cd comp_pipeline_restructure
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/check_pipeline_paths.py
```

**NCI Gadi:** batch compute nodes usually have **no outbound internet**. `pip install` then fails with `Network is unreachable` / `No matching distribution found for setuptools`. Do **`pip install -e .` on the login node** (or copy a pre-built venv from `$gdata`), and in PBS jobs only `source .venv/bin/activate` and run Python—no pip. For `scripts/hpc_login_smoke_test.sh`, set `export SKIP_PIP_INSTALL=1` inside the job after the venv exists.

**macOS → Linux: `pip` `UnicodeDecodeError` / `._*egg-info`**

Archives built or touched on macOS can add AppleDouble files (`._something`) next to real directories. If `cluster_completeness_pipeline.egg-info` (or a `._…egg-info` sibling) is present in the tree, pip may try to read it as UTF-8 metadata and crash.

On the HPC clone (before or after a failed `pip`):

```bash
cd comp_pipeline_restructure
find . -name '.DS_Store' -delete 2>/dev/null || true
find . -name '._*' -delete 2>/dev/null || true
rm -rf cluster_completeness_pipeline.egg-info .eggs
rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

New bundles from `scripts/package_for_hpc.sh` exclude `**/._*`, `*.egg-info`, and `__MACOSX` so this happens less often.

Then follow sections below for scheduler/job scripts.

**ngc1566 (clean slate + local one-liner + Gadi PBS example):** see [RUN_PIPELINE_NGC1566_LOCAL_AND_GADI.md](RUN_PIPELINE_NGC1566_LOCAL_AND_GADI.md).

---

## 1. Directory layout on HPC

Use a single **project directory** that contains both the code and the data the pipeline expects (so that `scripts/run_pipeline.py` can resolve paths like `ROOT/SLUG_library`, `ROOT/PSF_files`). For example:

```
$SCRATCH/cluster_completeness/     # or $PROJECT, $HOME, etc.
├── cluster-completeness-pipeline/   # git clone (or copy) of the repo
│   ├── scripts/
│   ├── cluster_pipeline/
│   ├── pyproject.toml
│   └── ...
├── SLUG_library/                   # SLUG cluster library (symlink or copy)
├── PSF_files/                      # PSF FITS (symlink or copy)
├── ngc628-c/                       # Galaxy FITS, config, readme (see RUNNING.md)
│   ├── ngc628-c_white.fits
│   ├── r2_wl_aa_ngc628-c.config
│   ├── automatic_catalog*_ngc628.readme
│   └── ...
├── galaxy_filter_dict.npy
├── output.param
├── default.nnw
├── .deps/
│   └── local/
│       └── bin/
│           └── bl                  # BAOlab executable
└── venv/                           # Python venv (or use conda)
```

Important: **run the pipeline from inside the repo directory** so that `ROOT` (derived from `scripts/run_pipeline.py` location) is the repo root. Then the code looks for `ROOT/SLUG_library`, `ROOT/PSF_files`, etc. So either:

- Put **data and config** inside the repo directory (e.g. `cluster-completeness-pipeline/SLUG_library`, `cluster-completeness-pipeline/ngc628-c`), or  
- Clone the repo and **symlink** `SLUG_library`, `PSF_files`, `ngc628-c`, etc. into the repo root.

Example (repo is `cluster-completeness-pipeline`, data lives in `$SCRATCH/data`):

```bash
cd $SCRATCH/cluster_completeness
git clone <your-repo-url> cluster-completeness-pipeline
cd cluster-completeness-pipeline
ln -s $SCRATCH/data/SLUG_library   SLUG_library
ln -s $SCRATCH/data/PSF_files      PSF_files
ln -s $SCRATCH/data/ngc628-c       ngc628-c
# copy or link galaxy_filter_dict.npy, output.param, default.nnw into repo root
```

---

## 2. Environment

- **Python 3.10+**: load via `module load python/3.11` (or use conda).
- **SExtractor**: `module load sextractor` or install under `.deps` and set `PATH`.
- **BAOlab**: build and place `bl` in `.deps/local/bin` (or set `COMP_BAO_PATH`).
- **IRAF** (only for 5-filter photometry): if available on HPC, set `IRAF` and ensure the `noao.digiphot.apphot` package is loaded (so tasks like `datapars` exist). When you use `--run_photometry`, Phase B runs **single-process** only (IRAF is not safe with multiple workers); `--parallel` is ignored. Otherwise run without `--run_photometry`.

Create a venv and install the package:

```bash
cd $SCRATCH/cluster_completeness/cluster-completeness-pipeline
module load python/3.11
python -m venv venv
source venv/bin/activate
pip install -e .
# If you need 5-filter photometry and IRAF:
# pip install pyraf
```

**提交前做一次路径/文件预检（guardrail）：**  
在登录节点设好与 PBS 相同的环境变量后，先运行预检，通过后再 `qsub`：

```bash
export COMP_SLUG_LIB_DIR=/path/to/SLUG_library
export COMP_PSF_PATH=/path/to/PSF_files
export COMP_FITS_PATH=/path/to/ngc628_data
export COMP_BAO_PATH=/path/to/.deps/local/bin
cd /path/to/your_project_root
module load python3/3.11.0
source .venv/bin/activate   # 若用 venv

# 只做预检，不跑 pipeline（exit 0 表示通过）
python scripts/check_pipeline_paths.py
# 若缺 LEGUS 星系目录（HLSP / white / r2），先自动下载再检查：
python scripts/check_pipeline_paths.py --galaxy ngc1566 --setup-if-missing
# 若跑 photometry，加上 5-filter 与 IRAF 检查：
python scripts/check_pipeline_paths.py --run-photometry
# 或通过 run_pipeline 调用：
python scripts/run_pipeline.py --check-only
python scripts/run_pipeline.py --check-only --run_photometry
```

预检会检查：main_dir、各 COMP_* 路径、galaxy_names.npy、galaxy_filter_dict.npy、output.param、default.nnw、BAOlab、PSF、SLUG 库、白光/科学帧、SExtractor 配置；加 `--run-photometry` 时还会检查各 filter 的 FITS、header_info、readme、IRAF。任一缺失会打印 `[MISS]` 并 exit 1。

Optional env vars (if paths differ from repo root):

```bash
export COMP_MAIN_DIR="$PWD"
export COMP_BAO_PATH="$PWD/.deps/local/bin"
export COMP_SLUG_LIB_DIR="$PWD/SLUG_library"
export COMP_PSF_PATH="$PWD/PSF_files"
```

---

## 3. SLURM job script (single-node run)

Save as `run_pipeline.slurm` (adjust partition, time, and paths):

```bash
#!/bin/bash
#SBATCH --job-name=comp_pipeline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Project dir: repo root (where SLUG_library, PSF_files, ngc628-c live)
WORKDIR=${WORKDIR:-$SCRATCH/cluster_completeness/cluster-completeness-pipeline}
cd "$WORKDIR" || exit 1

# Load modules (adjust to your cluster)
module purge
module load python/3.11

# Activate venv
source venv/bin/activate

# Optional: set path overrides
# export COMP_MAIN_DIR="$WORKDIR"
# export COMP_BAO_PATH="$WORKDIR/.deps/local/bin"
# export COMP_SLUG_LIB_DIR="$WORKDIR/SLUG_library"

# Run pipeline (no 5-filter photometry; omit --run_photometry if no IRAF)
python scripts/run_pipeline.py --cleanup --nframe 2 --ncl 500 --reff_list "1,3,6,10"

# With 5-filter photometry (requires IRAF and per-filter data):
# export IRAF=/path/to/iraf
# python scripts/run_pipeline.py --cleanup --nframe 2 --ncl 500 --reff_list "1,3,6,10" --run_photometry
```

Submit:

```bash
sbatch run_pipeline.slurm
```

---

## 4. PBS (Torque) example

```bash
#!/bin/bash
#PBS -N comp_pipeline
#PBS -l nodes=1:ppn=4
#PBS -l mem=16gb
#PBS -l walltime=08:00:00
#PBS -o slurm_%j.out
#PBS -e slurm_%j.err

WORKDIR=${WORKDIR:-$SCRATCH/cluster_completeness/cluster-completeness-pipeline}
cd "$WORKDIR" || exit 1

module load python/3.11
source venv/bin/activate

python scripts/run_pipeline.py --cleanup --nframe 2 --ncl 500 --reff_list "1,3,6,10"
```

Submit: `qsub run_pipeline.pbs`

---

## 4.1 Auto-submit GPU after CPU success

If you want GPU NN training to start automatically only when CPU pipeline succeeds:

1. Prepare CPU PBS script (for pipeline), e.g. `scripts/run_pipeline_pbs.sh`.
2. Prepare GPU PBS script (for NN), e.g. `scripts/run_nn_gpu_pbs.sh`.
3. Submit with dependency helper:

```bash
bash scripts/submit_cpu_then_gpu.sh
```

This internally does:

```bash
CPU_JOB_ID=$(qsub scripts/run_pipeline_pbs.sh)
qsub -W depend=afterok:${CPU_JOB_ID} scripts/run_nn_gpu_pbs.sh
```

So GPU job starts **only** when CPU job exits with code 0.

---

## 5. Later steps (ML inputs + NN training)

These can be separate jobs (no SExtractor/BAOlab/IRAF).

**Build ML inputs** (after pipeline has produced detection labels and physprop):

```bash
python scripts/build_ml_inputs.py --main-dir . --galaxy ngc628-c --outname test \
  --nframe 2 --reff-list 1 3 6 10 \
  --out-det det_3d.npy --out-npz allprop.npz
```

**Train NN** (optional GPU partition):

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
python scripts/perform_ml_to_learn_completeness.py \
  --det-path det_3d.npy --npz-path allprop.npz \
  --out-dir ./nn_sweep_out --save-best ...
```

---

## 6. Checklist

| Item | Action |
|------|--------|
| Repo and data on shared storage | Clone repo; put or symlink `SLUG_library`, `PSF_files`, `ngc628-c`, config files under repo root |
| Python 3.10+ | `module load` or conda; `pip install -e .` |
| SExtractor | In `PATH` or module |
| BAOlab | `.deps/local/bin/bl` or `COMP_BAO_PATH` |
| IRAF (optional) | Only for `--run_photometry`; set `IRAF`, load noao.digiphot.apphot. Phase B runs single-process when photometry is used. |
| Working directory | Always `cd` to repo root before `python scripts/run_pipeline.py` |
| Job resources | ~4–8 CPUs, 16–32 GB RAM, 4–8 h typical for nframe=2, ncl=500 |

---

## 7. References

- **Required files:** `docs/RUNNING.md`
- **Pipeline stages:** `docs/PIPELINE_FILES.md`
- **Scripts:** `docs/SCRIPTS.md`

---

## 8. Fastest run with 4×104-CPU queues (split by reff)

If you have **4 queues, each with 104 CPUs**, the fastest approach with **no code changes** is to **split the `reff_list` across 4 independent jobs**. Each job runs the full pipeline (Phase A + B) for its own subset of reff; all write to the **same directory** (shared filesystem). Output filenames include `reff`, so there is no overwrite.

### Strategy

- Example: `reff_list = "1,2,3,4,5,6,7,8,9,10"` and `nframe=2` → 20 (frame, reff) tasks.
- Split reff into 4 chunks, e.g.  
  Job 1: `--reff_list "1,2,3"`  
  Job 2: `--reff_list "4,5,6"`  
  Job 3: `--reff_list "7,8"`  
  Job 4: `--reff_list "9,10"`  
- Submit all 4 jobs **at once** (parallel). Each job uses one node; 104 CPUs per node are more than needed for the current (largely sequential) pipeline, so request e.g. **16–32 CPUs** and **1 node** per job to leave capacity for others, or use full 104 if you want to reserve the node.
- After **all 4 jobs finish**, run **build_ml_inputs** and **perform_ml_to_learn_completeness** once (same machine or a small follow-up job), passing the **full** reff list so they see all outputs.

**Single-node run:** If you are not splitting reff (one job with full reff list), you can add `--run_ml` to `run_pipeline.py` so it runs build_ml_inputs and perform_ml_to_learn_completeness automatically after the pipeline.

### Example: 4-way SLURM job array

Save as `run_pipeline_4way.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=comp_pipeline
#SBATCH --array=0-3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

WORKDIR=${WORKDIR:-$SCRATCH/cluster_completeness/cluster-completeness-pipeline}
cd "$WORKDIR" || exit 1

module purge
module load python/3.11
source venv/bin/activate

# Split reff_list into 4 chunks (adjust to your full list)
REFF_CHUNKS=("1,2,3" "4,5,6" "7,8" "9,10")
REFF_THIS_JOB=${REFF_CHUNKS[$SLURM_ARRAY_TASK_ID]}

echo "Job ${SLURM_ARRAY_TASK_ID}: reff_list=${REFF_THIS_JOB}"

python scripts/run_pipeline.py --cleanup --nframe 2 --ncl 500 --reff_list "$REFF_THIS_JOB"
```

Submit: `sbatch run_pipeline_4way.slurm`. After all 4 finish:

```bash
cd "$WORKDIR"
source venv/bin/activate
python scripts/build_ml_inputs.py --main-dir . --galaxy ngc628-c --outname test \
  --nframe 2 --reff-list 1 2 3 4 5 6 7 8 9 10 \
  --out-det det_3d.npy --out-npz allprop.npz
python scripts/perform_ml_to_learn_completeness.py --det-path det_3d.npy --npz-path allprop.npz \
  --out-dir ./nn_sweep_out --save-best ...
```

### Why this is fastest (with current code)

- **Phase A** (generate_white_clusters) runs once per job and only generates synthetic FITS for that job’s reff → no duplicate work.
- **Phase B** (detection + matching, and optionally photometry) runs per (frame, reff) inside each job; each job only handles its reff subset.
- **4 jobs in parallel** → wall-clock time is roughly that of **one** job (the slowest of the four), instead of running the full reff list sequentially on one node.
- Using **104 CPUs on one node** would only help if the pipeline itself parallelized over (frame, reff); with **`--parallel --n_workers 104`** Phase B does run (frame, reff) jobs in parallel, so one node can finish quickly for moderate job counts (see next section).

---

## 9. Example: 30k clusters per galaxy, 10 reff, 500 per frame (104 CPUs)

If you want **30,000 clusters per galaxy**, with **10 reff values** (e.g. 1–10 pc) and **500 clusters per frame**:

- Total (frame, reff) jobs = **nframe × 10**. To get 30,000 clusters:  
  `30,000 = nframe × 10 × 500` → **nframe = 6**.
- So **60 jobs** total (6 frames × 10 reff). Each job generates/detects 500 clusters.

### Commands

```bash
# From repo root, with venv activated
python scripts/run_pipeline.py --cleanup \
  --ncl 500 --nframe 6 --reff_list "1,2,3,4,5,6,7,8,9,10" \
  --parallel --n_workers 104
```

Phase A (`generate_white_clusters.py`) is invoked by `run_pipeline` with the same nframe/reff_list; it uses a process pool of size `min(60, cpu_count())`, so with 104 CPUs all 60 (frame, reff) synthetic frames are built in parallel. Phase B then runs 60 tasks in parallel with `n_workers=104`. **Do not** use `--run_photometry` with `--parallel`: 5-filter photometry uses IRAF and Phase B is forced to single-process when photometry is enabled.

### Rough wall-time (104 CPUs, one galaxy)

| Phase | What runs in parallel | Per-job estimate | Wall time (60 jobs, 104 CPUs) |
|-------|------------------------|-------------------|--------------------------------|
| **A** | 60 × `generate_white` (BAOlab + coords) | ~5–15 min (BAOlab/image size dependent) | **~5–15 min** |
| **B** | 60 × (copy + SExtractor + match) | ~2–6 min (SExtractor 1–5 min per FITS) | **~2–6 min** |
| **Total** | | | **~10–25 min to ~1 h** per galaxy |

So for **one galaxy** with this setup, expect about **15–60 minutes** wall time on a 104-CPU node, depending on FITS size and BAOlab/SExtractor speed. If Phase A or B is I/O or shared-filesystem bound, wall time can be higher.

### SLURM example (single node, 104 CPUs)

```bash
#!/bin/bash
#SBATCH --job-name=comp_30k
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

WORKDIR=${WORKDIR:-$SCRATCH/cluster_completeness/cluster-completeness-pipeline}
cd "$WORKDIR" || exit 1
source venv/bin/activate

python scripts/run_pipeline.py --cleanup \
  --ncl 500 --nframe 6 --reff_list "1,2,3,4,5,6,7,8,9,10" \
  --parallel --n_workers 104
```

---

## 10. Example: 300k clusters per galaxy, 10 reff, 500 per frame (104 CPUs)

If you want **300,000 clusters per galaxy**, still with **10 reff** and **500 per frame**:

- `300,000 = nframe × 10 × 500` → **nframe = 60**.
- So **600 jobs** total (60 frames × 10 reff). With **104 CPUs** you run 104 jobs at a time → **≈6 waves** (600/104).

### Commands

```bash
python scripts/run_pipeline.py --cleanup \
  --ncl 500 --nframe 60 --reff_list "1,2,3,4,5,6,7,8,9,10" \
  --parallel --n_workers 104
```

### Rough wall-time (104 CPUs, one galaxy)

| Phase | Jobs | Waves (104 CPUs) | Per-job | Wall time |
|-------|------|-------------------|---------|------------|
| **A** | 600 × `generate_white` | ~6 | ~5–15 min | **~30–90 min** |
| **B** | 600 × (copy + SExtractor + match) | ~6 | ~2–6 min | **~12–36 min** |
| **Total** | | | | **~45 min – 2 h** per galaxy |

Request at least **2–3 hours** in SLURM (e.g. `#SBATCH --time=03:00:00`) to allow for I/O and variance.

### SLURM example (300k clusters)

```bash
#!/bin/bash
#SBATCH --job-name=comp_300k
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

WORKDIR=${WORKDIR:-$SCRATCH/cluster_completeness/cluster-completeness-pipeline}
cd "$WORKDIR" || exit 1
source venv/bin/activate

python scripts/run_pipeline.py --cleanup \
  --ncl 500 --nframe 60 --reff_list "1,2,3,4,5,6,7,8,9,10" \
  --parallel --n_workers 104
```

---

## 11. Limited storage: what to do (no lock needed)

When running 300k clusters (600 jobs), **peak disk use** is:

- **Phase A:** 600 synthetic FITS in `white/synthetic_fits/` (each frame can be hundreds of MB to a few GB).
- **Phase B:** Each worker copies one FITS into its temp dir and runs SExtractor there. So you have **600 source FITS + (n_workers) temp copies** at once.

You do **not** need a lock for storage: each (frame_id, reff) is written and read by a single job; there is no shared writable file that multiple workers update. The only "contention" is total disk space.

### Options (pick one or combine)

1. **Limit concurrency**  
   Use fewer workers so fewer temp copies exist at once:
   ```bash
   --parallel --n_workers 20
   ```
   Peak storage = 600 FITS + 20 temp dirs. Run time goes up (more waves of jobs).

2. **Delete synthetic FITS after each job**  
   Use **`--delete_synthetic_after_use`**: after each (frame, reff) job finishes, the pipeline deletes that job's source synthetic FITS and `white_position_*.txt` file. Temp dir for that job is also removed (same as `keep_frames=False`).
   - **Peak** is unchanged during Phase B (all 600 FITS still exist at the start; they are removed only after the job that uses them completes).
   - **After the run**, `white/synthetic_fits/` is empty (and disk is freed).
   - Use this if you need to free space as soon as possible after the run, or if you run in "waves" (see below) so that only a subset of FITS exists at any time.

   Example:
   ```bash
   python scripts/run_pipeline.py --cleanup \
     --ncl 500 --nframe 60 --reff_list "1,2,3,4,5,6,7,8,9,10" \
     --parallel --n_workers 104 --delete_synthetic_after_use
   ```

3. **Wave-based run (manual)**  
   If you cannot hold 600 FITS at once:
   - Run Phase A for a **subset** of (frame, reff) (e.g. by splitting `reff_list` or `nframe` into chunks and calling `generate_white_clusters.py` with different args for each chunk).
   - Run Phase B for that subset with `--delete_synthetic_after_use`.
   - Repeat for the next chunk. So at any time you only have one chunk's FITS on disk. This requires generating and running in chunks (e.g. by `reff_list` as in section 8).

### Summary

| Goal | Action |
|------|--------|
| Lower peak temp dirs | `--n_workers 20` (or similar) |
| Free disk after run | `--delete_synthetic_after_use` |
| Never have all 600 FITS on disk | Run in waves (split reff/frames, Phase A + B + delete per chunk) |
| Lock for storage | **Not used** — no shared writable file; only disk space is the limit |

### Real run (Phase A + B, 2 frames × 2 reff, no 5-filter photometry)

A real run was executed: **cleanup → Phase A + Phase B** with `nframe=2`, `reff_list="1,3"`, `ncl=500`, **no** `--run_photometry`.  
White images are **7200×7200** (LEGUS). Measured:

| Item | This run (4 jobs) | Extrapolated to 300k (600 jobs) |
|------|-------------------|----------------------------------|
| white/synthetic_fits | ~791 MB | ~116 GB |
| white/baolab (Phase A intermediates) | (included in run) | scales with n_jobs |
| white_position_*.txt + physprop | ~0.5 MB | ~75 MB |
| white (matched_coords, diagnostics, labels, catalogue) | ~127 KB | ~19 MB |
| **tmp_pipeline_test** (2 workers) | **~1.55 GB** | **~80 GB** (104 workers) |
| **TOTAL** | **~2.3 GB** | **~196 GB** |

With **5-filter photometry** enabled, add **galaxy/*/synthetic_fits** (one FITS per filter per job) and **galaxy/*/photometry**; expect roughly **several hundred GB more** (e.g. 5× filter synthetic FITS of similar size to white).

To reproduce the measurement (after installing SLUG, BAOlab, SExtractor, ngc628-c data):

```bash
python scripts/run_real_storage_estimate.py --nframe 2 --reff_list "1,3" --no_photometry
# With 5-filter: drop --no_photometry (adds 5× synthetic_fits + photometry)
```

### Real run with 5-filter photometry (2 frames × 2 reff)

Same setup with **`--run_photometry`** (Phase B runs inject_clusters_to_5filters + photometry + catalogue). Measured:

| Item | This run (4 jobs) | Extrapolated to 300k (600 jobs) |
|------|-------------------|----------------------------------|
| white/synthetic_fits | ~791 MB | — |
| white/baolab | ~2.44 GB | — |
| white (matched_coords, diagnostics, labels, catalogue) | ~397 KB | ~60 MB |
| physprop | ~143 KB | ~21 MB |
| tmp_pipeline_test | ~1.55 GB | ~80 GB (104 workers) |
| **galaxy/*/synthetic_fits (5 filters)** | **~3.86 GB** | **~579 GB** |
| galaxy/*/photometry (5 filters) | ~0 KB | ~0 KB |
| **TOTAL** | **~8.62 GB** | **~1.1–1.5 TB** |

So with 5-filter photometry, **one galaxy 300k run** needs on the order of **1–1.5 TB** free (white + baolab + 5× filter synthetic FITS + tmp). Use `--delete_synthetic_after_use` and/or fewer workers to reduce peak.

### Measuring storage (dummy FITS, no pipeline run)

Use **`scripts/estimate_storage.py`** to create dummy FITS and coords, measure sizes, and extrapolate to 300k clusters:

```bash
# Test grid: 3 frames × 2 reff = 6 jobs; extrapolate to 300,000 clusters (60×10=600 jobs)
python scripts/estimate_storage.py --nframe 3 --nreff 2 --ncl 500 --extrapolate 300000 --n_workers 104
```

Example output (4k×4k white image, float32):

- **Per (frame, reff):** ~64 MB (one FITS) + ~11 KB (white_position).
- **Phase A total for 600 jobs:** ~37.5 GB.
- **Phase B peak (600 FITS + 104 temp dirs):** ~44 GB.

For **8k×8k** white images, Phase A ≈ 150 GB and Phase B peak ≈ 176 GB. Ensure at least that much free space (or run with `--n_workers 20` and/or `--delete_synthetic_after_use` to reduce peak).
