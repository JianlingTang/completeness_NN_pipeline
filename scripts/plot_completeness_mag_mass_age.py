#!/usr/bin/env python3
"""
Plot completeness vs mag / log(mass) / log(age) (same as integration test figure).
Saves to tests/output/completeness_mag_mass_age_bins.png so you can open the figure.

Usage:
  python scripts/plot_completeness_mag_mass_age.py
  python scripts/plot_completeness_mag_mass_age.py --out tests/output/my_completeness.png

Run from repo root so cluster_pipeline is importable, or set PYTHONPATH to repo root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cluster_pipeline.pipeline.diagnostics import completeness_per_bin


def binned_completeness_percentile(x, y_det, n_bins=15):
    ok = np.isfinite(x)
    x_ok, y_ok = x[ok], y_det[ok]
    if len(x_ok) < 2:
        return np.array([]), np.array([]), np.array([])
    bins = np.percentile(x_ok, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return np.array([np.median(x_ok)]), np.array([np.mean(y_ok)]), np.array([len(x_ok)])
    hist_total, _ = np.histogram(x_ok, bins=bins)
    hist_det, _ = np.histogram(x_ok, bins=bins, weights=y_ok)
    centers = (bins[:-1] + bins[1:]) / 2
    comp = np.where(hist_total > 0, hist_det / hist_total.astype(float), np.nan)
    return centers, comp, hist_total


def main():
    ap = argparse.ArgumentParser(description="Plot completeness vs mag / mass / age (synthetic data)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "tests" / "output" / "completeness_mag_mass_age_bins.png")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required", file=sys.stderr)
        sys.exit(1)

    np.random.seed(99)
    n = 500
    mags = np.random.uniform(18, 26, n)
    p_mag = 1.0 / (1.0 + np.exp(-1.2 * (22.0 - mags)))
    matched_mag = (np.random.rand(n) < p_mag).astype(np.int8)

    np.random.seed(100)
    log_mass = np.random.uniform(5, 8, n)
    p_mass = 1.0 / (1.0 + np.exp(-1.5 * (log_mass - 6.5)))
    matched_mass = (np.random.rand(n) < p_mass).astype(np.int8)

    np.random.seed(101)
    log_age = np.random.uniform(6, 9, 400)
    p_age = 1.0 / (1.0 + np.exp(-0.8 * (log_age - 7.5)))
    matched_age = (np.random.rand(400) < p_age).astype(np.int8)

    cx_mag, cy_mag, _ = completeness_per_bin(mags, matched_mag, mag_min=17, mag_max=27, n_bins=15)
    cx_mass, cy_mass, _ = binned_completeness_percentile(log_mass, matched_mass, n_bins=15)
    cx_age, cy_age, _ = binned_completeness_percentile(log_age, matched_age, n_bins=12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(cx_mag, cy_mag, "o-")
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Completeness")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Completeness vs mag (synthetic logistic)")
    axes[0].grid(True, alpha=0.3)

    if len(cx_mass) > 0:
        axes[1].plot(cx_mass, cy_mass, "o-", color="C1")
    axes[1].set_xlabel(r"$\log_{10}$(mass)")
    axes[1].set_ylabel("Completeness")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Completeness vs mass (synthetic)")
    axes[1].grid(True, alpha=0.3)

    if len(cx_age) > 0:
        axes[2].plot(cx_age, cy_age, "o-", color="C2")
    axes[2].set_xlabel(r"$\log_{10}$(age)")
    axes[2].set_ylabel("Completeness")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Completeness vs age (synthetic)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Pipeline completeness visualization (synthetic logistic data)")
    fig.tight_layout()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
