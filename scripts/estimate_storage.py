#!/usr/bin/env python3
"""
Estimate disk storage for the completeness pipeline from a small test run.

Creates dummy synthetic FITS and coords of typical size, measures disk usage,
then extrapolates to 300,000 clusters (60 frames × 10 reff × 500 cl/frame).

Usage:
    python scripts/estimate_storage.py [--nframe 3] [--nreff 2] [--ncl 500]
    python scripts/estimate_storage.py --extrapolate 300000

Output: per-(frame,reff) size, total for test grid, and extrapolated for 300k.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Estimate pipeline storage from dummy FITS + coords")
    ap.add_argument("--nframe", type=int, default=3, help="Number of frames for test (default 3)")
    ap.add_argument("--nreff", type=int, default=2, help="Number of reff values for test (default 2)")
    ap.add_argument("--ncl", type=int, default=500, help="Clusters per frame (default 500)")
    ap.add_argument("--width", type=int, default=4096, help="FITS NAXIS1 (default 4096)")
    ap.add_argument("--height", type=int, default=4096, help="FITS NAXIS2 (default 4096)")
    ap.add_argument("--extrapolate", type=int, default=300_000, help="Extrapolate to this many clusters (default 300000)")
    ap.add_argument("--n_workers", type=int, default=104, help="Workers for Phase B peak temp (default 104)")
    args = ap.parse_args()

    try:
        from astropy.io import fits
    except ImportError:
        print("astropy required: pip install astropy")
        return 1

    nframe, nreff, ncl = args.nframe, args.nreff, args.ncl
    h, w = args.height, args.width
    n_jobs_test = nframe * nreff
    # 300k = nframe * 10 * 500  => nframe = 60
    total_clusters_target = args.extrapolate
    n_reff_full = 10
    nframe_full = total_clusters_target // (n_reff_full * args.ncl)  # 60 for 300k
    n_jobs_full = nframe_full * n_reff_full

    with tempfile.TemporaryDirectory(prefix="storage_est_") as tmp:
        tmp = Path(tmp)
        # One synthetic FITS (typical white-light LEGUS: float32)
        arr = np.zeros((h, w), dtype=np.float32)
        fits_path = tmp / "test_frame.fits"
        fits.HDUList([fits.PrimaryHDU(arr)]).writeto(fits_path, overwrite=True)
        size_fits = fits_path.stat().st_size

        # One white_position file (x y mag, ~25-40 chars/line, same as legus_original_pipeline)
        coords_path = tmp / "white_position_0_test_reff1.00.txt"
        with open(coords_path, "w") as f:
            for i in range(ncl):
                f.write(f"  {100.0 + i}  {200.0 + i}  {18.5 + i * 0.01}\n")
        size_coords = coords_path.stat().st_size

        # Approximate SExtractor output per frame (catalog + .coo): ~few MB for 500 sources
        size_sextractor_per_frame = 2 * 1024 * 1024

    # Per (frame, reff) in synthetic_fits
    per_job_synth = size_fits + size_coords
    total_test_synth = n_jobs_test * size_fits + n_jobs_test * size_coords
    # Phase B peak: all synth FITS + n_workers temp copies (FITS + SExtractor)
    per_worker_temp = size_fits + size_sextractor_per_frame
    peak_phase_b_test = n_jobs_test * size_fits + min(args.n_workers, n_jobs_test) * per_worker_temp
    peak_phase_b_full = n_jobs_full * size_fits + min(args.n_workers, n_jobs_full) * per_worker_temp

    def fmt(b):
        if b >= 1024**3:
            return f"{b / 1024**3:.2f} GB"
        if b >= 1024**2:
            return f"{b / 1024**2:.2f} MB"
        return f"{b / 1024:.2f} KB"

    print("=" * 60)
    print("Storage estimate (dummy FITS + coords)")
    print("=" * 60)
    print(f"  FITS size ({w}x{h} float32):     {fmt(size_fits)}")
    print(f"  white_position ({ncl} rows):    {fmt(size_coords)}")
    print(f"  Per (frame, reff) synth:        {fmt(per_job_synth)}")
    print()
    print(f"  Test grid: nframe={nframe}, nreff={nreff}  =>  {n_jobs_test} jobs")
    print(f"  Phase A total (synthetic_fits + coords):   {fmt(total_test_synth)}")
    print(f"  Phase B peak (synth + {args.n_workers} temp dirs):  {fmt(peak_phase_b_test)}")
    print()
    print("  Extrapolation to {} clusters (nframe={}, nreff={}, {} jobs):".format(
        total_clusters_target, nframe_full, n_reff_full, n_jobs_full))
    total_synth_full = n_jobs_full * (size_fits + size_coords)
    print(f"  Phase A total:                             {fmt(total_synth_full)}")
    print(f"  Phase B peak (synth + {args.n_workers} workers):        {fmt(peak_phase_b_full)}")
    print()
    print("  Recommendation: ensure at least {} free for a single-galaxy 300k run.".format(
        fmt(peak_phase_b_full)))
    print("  Use --delete_synthetic_after_use to free synthetic FITS after each job;")
    print("  use --n_workers 20 to reduce peak temp dirs if needed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
