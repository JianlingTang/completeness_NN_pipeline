#!/usr/bin/env python3
"""
Run a real Phase A + Phase B (with 5-filter photometry) and measure disk usage,
then extrapolate to 300,000 clusters.

Step 1: Delete all previous pipeline outputs (cleanup).
Step 2: Run Phase A + Phase B with small params (nframe, nreff).
Step 3: Measure storage of white/synthetic_fits, 5-filter synthetic_fits,
        tmp dirs, physprop, matched_coords, catalogue, detection_labels, etc.
Step 4: Extrapolate to 300k (60 frames × 10 reff).

Usage:
    python scripts/run_real_storage_estimate.py --nframe 2 --reff_list "1,3" [--run_photometry]
    python scripts/run_real_storage_estimate.py   # defaults: 2 frames, reff 1,3, with photometry

Requires: SLUG library, BAOlab, ngc628-c data, SExtractor. Optional: IRAF for photometry.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GALAXY = "ngc628-c"
OUTNAME = "test"
NCL = 500
# 300k = 60 * 10 * 500
NFRAME_300K = 60
NREFF_FULL = 10


def dir_size(path: Path) -> int:
    """Total size of path (file or directory tree) in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total


def fmt(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    return f"{b / 1024:.2f} KB"


def main():
    ap = argparse.ArgumentParser(description="Real Phase A+B run + storage measure + extrapolate to 300k")
    ap.add_argument("--nframe", type=int, default=2, help="Number of frames (default 2)")
    ap.add_argument("--reff_list", type=str, default="1,3", help="Comma-separated reff (default 1,3)")
    ap.add_argument("--ncl", type=int, default=NCL, help="Clusters per frame (default 500)")
    ap.add_argument("--run_photometry", action="store_true", default=True, help="Run 5-filter photometry (default True)")
    ap.add_argument("--no_photometry", action="store_false", dest="run_photometry", help="Skip 5-filter photometry")
    ap.add_argument("--n_workers", type=int, default=4, help="Parallel workers for Phase B (default 4)")
    ap.add_argument("--extrapolate", type=int, default=300_000, help="Extrapolate to this many clusters (default 300000)")
    args = ap.parse_args()

    reff_list = [float(x.strip()) for x in args.reff_list.split(",") if x.strip()]
    nframe, nreff = args.nframe, len(reff_list)
    n_jobs = nframe * nreff
    n_jobs_300k = NFRAME_300K * NREFF_FULL

    # Step 1: Cleanup only (no Phase A/B yet)
    print("=" * 60)
    print("Step 1: Cleanup all pipeline outputs")
    print("=" * 60)
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
    run_pipeline_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_pipeline_mod)
    run_pipeline_mod.cleanup_all_pipeline_outputs()

    # Step 2: Run Phase A + Phase B (no --cleanup so we keep outputs for measurement)
    print("\n" + "=" * 60)
    print(f"Step 2: Run Phase A + B (nframe={nframe}, reff={reff_list}, ncl={args.ncl}, photometry={args.run_photometry})")
    print("=" * 60)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--nframe", str(nframe),
        "--reff_list", args.reff_list,
        "--ncl", str(args.ncl),
        "--parallel",
        "--n_workers", str(args.n_workers),
    ]
    if args.run_photometry:
        cmd.append("--run_photometry")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("\nPipeline failed (exit {}). Storage report below may be partial or empty.".format(result.returncode))
    else:
        print("\nPipeline completed successfully.")

    # Step 3: Measure storage
    print("\n" + "=" * 60)
    print("Step 3: Measured storage (after run)")
    print("=" * 60)

    white_dir = ROOT / GALAXY / "white"
    gal_dir = ROOT / GALAXY
    tmp_base = ROOT / "tmp_pipeline_test"
    physprop_dir = ROOT / "physprop"

    categories = []
    total_bytes = 0

    # White
    for name, label in [
        ("synthetic_fits", "white/synthetic_fits"),
        ("baolab", "white/baolab"),
        ("matched_coords", "white/matched_coords"),
        ("diagnostics", "white/diagnostics"),
        ("detection_labels", "white/detection_labels"),
        ("catalogue", "white/catalogue"),
        ("s_extraction", "white/s_extraction"),
    ]:
        d = white_dir / name
        sz = dir_size(d)
        categories.append((label, sz))
        total_bytes += sz
    # white_position_*.txt
    coord_files = list(white_dir.glob("white_position_*.txt")) + list(white_dir.glob("*_position_*_test_*.txt"))
    coord_size = sum(f.stat().st_size for f in coord_files if f.is_file())
    categories.append(("white_position_*.txt", coord_size))
    total_bytes += coord_size

    # Physprop
    sz = dir_size(physprop_dir)
    categories.append(("physprop/", sz))
    total_bytes += sz

    # Temp (Phase B workers)
    sz = dir_size(tmp_base)
    categories.append(("tmp_pipeline_test/", sz))
    total_bytes += sz

    # Per-filter: synthetic_fits + photometry
    filter_synth_total = 0
    filter_phot_total = 0
    if gal_dir.exists():
        for sub in sorted(gal_dir.iterdir()):
            if sub.is_dir() and sub.name != "white":
                s_synth = dir_size(sub / "synthetic_fits")
                s_phot = dir_size(sub / "photometry")
                filter_synth_total += s_synth
                filter_phot_total += s_phot
    categories.append(("galaxy/*/synthetic_fits (5 filters)", filter_synth_total))
    categories.append(("galaxy/*/photometry (5 filters)", filter_phot_total))
    total_bytes += filter_synth_total + filter_phot_total

    for label, sz in categories:
        print(f"  {label:<45} {fmt(sz):>12}")
    print(f"  {'TOTAL (this run)':<45} {fmt(total_bytes):>12}")

    # Step 4: Extrapolate to 300k
    print("\n" + "=" * 60)
    print("Step 4: Extrapolation to {} clusters (nframe={}, nreff={}, {} jobs)".format(
        args.extrapolate, NFRAME_300K, NREFF_FULL, n_jobs_300k))
    print("=" * 60)

    if n_jobs == 0:
        print("  No jobs in test run; cannot extrapolate.")
        return 1

    # Scale factors: some scale with n_jobs, temp scales with min(n_workers, n_jobs)
    scale_jobs = n_jobs_300k / n_jobs
    # White synthetic_fits + coords scale with n_jobs (baolab is intermediate, similar size to synthetic_fits)
    white_synth_measured = dir_size(white_dir / "synthetic_fits") + dir_size(white_dir / "baolab") + coord_size
    white_synth_300k = int(white_synth_measured * scale_jobs) if white_synth_measured else 0
    # Physprop scales with n_jobs
    phys_measured = dir_size(physprop_dir)
    phys_300k = int(phys_measured * scale_jobs) if phys_measured else 0
    # 5-filter synthetic_fits scale with n_jobs
    filter_synth_300k = int(filter_synth_total * scale_jobs) if filter_synth_total else 0
    # Photometry output (parquet, etc.) scales with n_jobs
    filter_phot_300k = int(filter_phot_total * scale_jobs) if filter_phot_total else 0
    # matched_coords, diagnostics, detection_labels, catalogue scale with n_jobs (exclude synthetic_fits and baolab, already in white_synth)
    other_white = sum(
        sz for _, sz in categories
        if "white/" in _ and "synthetic" not in _ and "position" not in _ and "baolab" not in _
    )
    other_white_300k = int(other_white * scale_jobs)
    # Temp: peak is n_workers * per_job_temp; per_job_temp ~ one FITS copy + SExtractor. So scale by n_workers (same) but more jobs so more concurrent temp dirs. For 600 jobs and 104 workers, peak temp ≈ 104 * per_job. We measured current run's tmp; if we had 4 workers and 4 jobs, tmp ≈ 4 * per_job. So temp_300k ≈ (104/4) * current_tmp if 104>4 else current_tmp * (n_jobs_300k/n_jobs). Simplified: temp scales with number of concurrent workers, so keep n_workers same → temp_300k = current_tmp * (min(104,n_jobs_300k) / min(4,n_jobs)).
    workers_test = min(args.n_workers, n_jobs)
    workers_300k = min(104, n_jobs_300k)
    tmp_measured = dir_size(tmp_base)
    tmp_300k = int(tmp_measured * (workers_300k / workers_test)) if workers_test and tmp_measured else 0

    total_300k = white_synth_300k + phys_300k + filter_synth_300k + filter_phot_300k + other_white_300k + tmp_300k
    print(f"  white synthetic_fits + coords       {fmt(white_synth_300k):>12}")
    print(f"  physprop                            {fmt(phys_300k):>12}")
    print(f"  galaxy/*/synthetic_fits (5 filt)    {fmt(filter_synth_300k):>12}")
    print(f"  galaxy/*/photometry (5 filt)       {fmt(filter_phot_300k):>12}")
    print(f"  white (matched, diag, labels, cat)  {fmt(other_white_300k):>12}")
    print(f"  tmp_pipeline_test (peak)           {fmt(tmp_300k):>12}")
    print(f"  {'EST. TOTAL (300k clusters)':<45} {fmt(total_300k):>12}")
    print()
    print("  Recommendation: ensure at least {} free for a single-galaxy 300k run.".format(fmt(total_300k)))
    print("  Use --delete_synthetic_after_use and/or --n_workers 20 to reduce peak.")
    print("=" * 60)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
