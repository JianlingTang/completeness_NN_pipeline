#!/usr/bin/env python3
"""
Sample N clusters from the SLUG library, compute their white-light magnitude
(mag_BAO) using the same formula as generate_white_clusters.py, and write
x y mag to a file suitable for BAOlab / --input_coords.

Usage:
    python scripts/sample_slug_white_mag.py [--n 20] [--seed 42] [--out FILE]

Output format: one line per cluster, "x y mag" (space-separated).
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Distance
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from slugpy import read_cluster  # noqa: E402
except ImportError:
    from cluster_pipeline.data.slug_reader import read_cluster  # noqa: E402


def generate_white_light(scale_factors, f275, f336, f438, f555, f814):
    """Same as in generate_white_clusters.py."""
    scale_factors = np.asarray(scale_factors) / scale_factors[2]
    denom = np.sqrt(np.sum((1.0 / scale_factors) ** 2))
    white = (
        np.sqrt(
            (f275 / scale_factors[0]) ** 2
            + (f336 / scale_factors[1]) ** 2
            + (f438 / scale_factors[2]) ** 2
            + (f555 / scale_factors[3]) ** 2
            + (f814 / scale_factors[4]) ** 2
        )
        / denom
    )
    return white


def main():
    ap = argparse.ArgumentParser(description="Sample SLUG clusters and compute white-light mag for BAOlab")
    ap.add_argument("--n", type=int, default=20, help="Number of clusters to sample")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--galaxy", type=str, default="ngc628-c", help="Galaxy id (for PHOTFLAM and frame)")
    ap.add_argument("--dmod", type=float, default=29.98, help="Distance modulus")
    ap.add_argument("--out", type=str, default=None, help="Output path (default: ngc628-c/white/slug_sampled_<n>_coords.txt)")
    ap.add_argument("--with_phys", action="store_true", help="Output 5 columns: x y mag mass age (same row order as pipeline clusters)")
    ap.add_argument("--fits_path", type=str, default=None, help="Path to FITS dir (default: ROOT)")
    args = ap.parse_args()

    fits_path = Path(args.fits_path or ROOT)
    gal = args.galaxy
    dmod = args.dmod
    n = args.n

    # 1. Galaxy filter order (same as generate_white_clusters)
    gal_filters = np.load(ROOT / "galaxy_filter_dict.npy", allow_pickle=True).item()
    filters = gal_filters[gal]
    filter_names_sorted = sorted(filters[0])  # e.g. ['f275w', 'f336w', 'f435w', 'f555w', 'f814w']
    # Build CAM_FILTER names for SLUG (match load_slug_libraries order)
    allfilters_cam = []
    for filt, cam in zip(filters[0], filters[1]):
        filt, cam = filt.upper(), cam.upper()
        if cam == "WFC3":
            cam = "WFC3_UVIS"
        allfilters_cam.append(f"{cam}_{filt}")
    allfilters_cam = sorted(allfilters_cam, key=lambda x: x[-4:])  # F275W, F336W, F435W, F555W, F814W

    # 2. Load SLUG library (L_lambda)
    libdir = ROOT / "SLUG_library"
    libname = str(libdir / "flat_in_logm")
    print(f"Loading SLUG library: {libname}")
    lib = read_cluster(libname, read_filters=allfilters_cam, photsystem="L_lambda")
    phot_neb_ex = lib.phot_neb_ex
    n_total = phot_neb_ex.shape[0]

    # 3. PHOTFLAM from galaxy FITS headers (same order as filter_names_sorted)
    headers = {}
    for fn in filter_names_sorted:
        pat = str(fits_path / gal / f"*{fn}*drc.fits")
        matches = glob.glob(pat)
        if not matches:
            raise FileNotFoundError(f"No FITS for {fn}: {pat}")
        _, headers[fn] = fits.getdata(matches[0], header=True)
    scaling_factors = np.array([headers[fn]["PHOTFLAM"] for fn in filter_names_sorted])

    # 4. Distance and flux conversion
    d = Distance(distmod=dmod * u.mag)
    distance_in_cm = d.to(u.cm).value
    phot_f = phot_neb_ex / (4 * np.pi * distance_in_cm ** 2)
    scaled_dataset = phot_f / scaling_factors

    f275 = scaled_dataset[:, 0]
    f336 = scaled_dataset[:, 1]
    f438 = scaled_dataset[:, 2]  # F435W column for this galaxy
    f555 = scaled_dataset[:, 3]
    f814 = scaled_dataset[:, 4]

    img21ms = generate_white_light([55.8, 45.3, 44.2, 65.7, 29.3], f275, f336, f438, f555, f814)
    img24rgb = generate_white_light([0.5, 0.8, 3.1, 11.2, 13.7], f275, f336, f438, f555, f814)
    slug_cluster_white_flux = 0.5 * (img21ms + img24rgb)
    mag_bao = np.log10(np.maximum(slug_cluster_white_flux, 1e-300)) / -0.4

    # 5. Random cluster indices
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_total, size=n, replace=False)
    mag_selected = mag_bao[indices]
    if args.with_phys:
        mass_selected = np.asarray(lib.actual_mass, dtype=float)[indices]
        age_selected = (np.asarray(lib.time, dtype=float) - np.asarray(lib.form_time, dtype=float))[indices]
        age_selected = np.maximum(age_selected, 1e6)

    # 6. Random (x, y) on science frame (valid pixels)
    white_fits = fits_path / gal / f"{gal}_white.fits"
    if not white_fits.exists():
        raise FileNotFoundError(f"Science frame not found: {white_fits}")
    sci = fits.getdata(white_fits)
    valid = np.where(sci.flatten() > 0)
    valid_flat = valid[0]
    if len(valid_flat) == 0:
        valid_flat = np.arange(sci.size)
    chosen_flat = rng.choice(valid_flat, size=n, replace=True)
    yy, xx = np.unravel_index(chosen_flat, sci.shape)
    # BAOlab / white_position format: x y mag (x = column, y = row)
    x_coords = xx.astype(int)
    y_coords = yy.astype(int)

    # 7. Output file
    out_path = args.out or str(ROOT / gal / "white" / f"slug_sampled_{n}_coords.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        if args.with_phys:
            for x, y, mag, mass, age in zip(x_coords, y_coords, mag_selected, mass_selected, age_selected):
                f.write(f"{x} {y} {mag} {mass} {age}\n")
        else:
            for x, y, mag in zip(x_coords, y_coords, mag_selected):
                f.write(f"{x} {y} {mag}\n")

    print(f"Wrote {n} lines to {out_path}" + (" (5 cols: x y mag mass age)" if args.with_phys else " (3 cols: x y mag)"))
    print("\n--- First 5 and last 2 lines (x y mag) ---")
    with open(out_path) as f:
        lines = f.readlines()
    for line in lines[:5]:
        print(" ", line.strip())
    if len(lines) > 7:
        print(" ...")
    for line in lines[-2:]:
        print(" ", line.strip())
    print("\n--- mag_bao stats (selected) ---")
    print(f"  min={mag_selected.min():.4f}  max={mag_selected.max():.4f}  mean={mag_selected.mean():.4f}")
    if args.with_phys:
        print("--- mass/age stats (selected) ---")
        print(f"  log10(mass): min={np.log10(mass_selected.min()):.3f}  max={np.log10(mass_selected.max()):.3f}")
        print(f"  log10(age/yr): min={np.log10(age_selected.min()):.3f}  max={np.log10(age_selected.max()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
