#!/usr/bin/env python3
"""
Plot completeness vs mass / age / mag for each catalogue criterion step:

- white detection only (white_match)
- + CI (passes_ci)
- + 4-band merr (>=4 filters with merr <= merr_cut)
- + M_V (full in_catalogue)

This lets us see at which stage the downturn (completeness drop) appears.

Usage (inside project root):

    python scripts/plot_criteria_diagnostics.py --nframe 4 --reff 9

Assumptions:
- Galaxy fixed to ngc628-c
- outname fixed to 'test'
- physprop, detection_labels, and catalogue outputs already exist
  (i.e. run scripts/run_pipeline.py first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GALAXY = "ngc628-c"
OUTNAME = "test"
DMOD = 29.98  # distance modulus used for NGC 628c (apparent = absolute + DMOD)


def binned_completeness_mag(mag: np.ndarray, y_det: np.ndarray, n_bins: int = 30):
    """
    Completeness per mag bin = (recovered in bin) / (injected in bin).
    mag, y_det aligned: one entry per injected cluster; y_det = 0/1.
    """
    ok = np.isfinite(mag)
    mag_ok, y_ok = mag[ok], y_det[ok]
    if len(mag_ok) < 2:
        return np.array([]), np.array([]), np.array([])
    lo, hi = np.nanmin(mag_ok), np.nanmax(mag_ok)
    if lo >= hi:
        return np.array([lo]), np.array([np.mean(y_ok)]), np.array([len(mag_ok)])
    bins = np.linspace(lo, hi, n_bins + 1)
    hist_total, _ = np.histogram(mag_ok, bins=bins)  # injected in bin
    hist_det, _ = np.histogram(mag_ok, bins=bins, weights=y_ok)  # recovered in bin
    centers = (bins[:-1] + bins[1:]) / 2
    comp = np.full_like(hist_total, np.nan, dtype=float)
    np.divide(hist_det, hist_total, out=comp, where=hist_total > 0)
    return centers, comp, hist_total


def load_physprop(main_dir: Path, reff: float, nframe: int):
    physprop_dir = main_dir / "physprop"
    mrmodel = "flat"
    masses = []
    ages = []
    mag5 = []  # (N,5)
    for i_frame in range(nframe):
        mass_path = physprop_dir / f"mass_select_model{mrmodel}_frame{i_frame}_reff{int(reff)}_{OUTNAME}.npy"
        age_path = physprop_dir / f"age_select_model{mrmodel}_frame{i_frame}_reff{int(reff)}_{OUTNAME}.npy"
        mag_path = physprop_dir / f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_reff{int(reff)}_{OUTNAME}.npy"
        if not (mass_path.exists() and age_path.exists() and mag_path.exists()):
            continue
        m = np.load(mass_path)
        a = np.load(age_path)
        mag = np.load(mag_path)
        if mag.ndim == 1:
            mag = mag.reshape(-1, 1)
        n_use = min(len(m), len(a), len(mag))
        masses.append(m[:n_use])
        ages.append(a[:n_use])
        mag5.append(mag[:n_use, :5])
    if not masses:
        raise RuntimeError("No physprop arrays found; run scripts/run_pipeline.py first.")
    mass = np.concatenate(masses)
    age = np.concatenate(ages)
    # mag_VEGA_select arrays are already in apparent Vega mag (SLUG mags + dmod in generate_white_clusters)
    mag5_all = np.concatenate(mag5, axis=0)
    return mass, age, mag5_all


def load_labels(main_dir: Path, reff: float, nframe: int):
    white_dir = main_dir / GALAXY / "white"
    labels_dir = white_dir / "detection_labels"
    cat_dir = white_dir / "catalogue"
    match_dir = cat_dir

    white_labels = []
    ci_flags = []
    merr_flags = []
    mv_flags = []
    cat_flags = []

    for i_frame in range(nframe):
        # white detection labels (per injected cluster, 0/1)
        white_path = labels_dir / f"detection_labels_white_match_frame{i_frame}_{OUTNAME}_reff{reff:.2f}.npy"
        det_path = labels_dir / f"detection_frame{i_frame}_{OUTNAME}_reff{reff:.2f}.npy"
        cat_path = cat_dir / f"catalogue_frame{i_frame}_{OUTNAME}_reff{reff:.2f}.parquet"
        match_path = match_dir / f"match_results_frame{i_frame}_{OUTNAME}_reff{reff:.2f}.parquet"
        if not (white_path.exists() and det_path.exists() and cat_path.exists() and match_path.exists()):
            continue
        wl = np.load(white_path)
        det = np.load(det_path)
        cat_df = pd.read_parquet(cat_path)
        match_df = pd.read_parquet(match_path)

        # Canonical order: match_results.cluster_id, which was used to build detection_frame*.npy
        order = match_df["cluster_id"].values
        cid_to_ci = dict(zip(cat_df["cluster_id"], cat_df["passes_ci"]))
        cid_to_merr = dict(zip(cat_df["cluster_id"], cat_df["passes_stage2_merr"]))
        cid_to_mv = dict(zip(cat_df["cluster_id"], cat_df["passes_MV"]))

        # detection labels (det) and white-match labels are aligned with match_result.detection_labels,
        # which use the same cluster_id order as match_results parquet.
        ci_arr = np.array([cid_to_ci.get(cid, 0) for cid in order], dtype=np.uint8)
        merr_arr = np.array([cid_to_merr.get(cid, 0) for cid in order], dtype=np.uint8)
        mv_arr = np.array([cid_to_mv.get(cid, 0) for cid in order], dtype=np.uint8)

        n_use = min(len(order), len(wl), len(det), len(ci_arr), len(merr_arr), len(mv_arr))
        white_labels.append(wl[:n_use])
        ci_flags.append(ci_arr[:n_use])
        merr_flags.append(merr_arr[:n_use])
        mv_flags.append(mv_arr[:n_use])
        cat_flags.append(det[:n_use])

    if not white_labels:
        raise RuntimeError("No detection_labels / catalogue parquet found for requested reff/nframe.")

    wl_all = np.concatenate(white_labels).astype(np.uint8)
    ci_all = np.concatenate(ci_flags).astype(np.uint8)
    merr_all = np.concatenate(merr_flags).astype(np.uint8)
    cat_all = np.concatenate(cat_flags).astype(np.uint8)

    # Labels for each criterion step (0/1)
    l_white = wl_all
    l_ci = wl_all & ci_all
    l_merr = wl_all & ci_all & merr_all
    l_cat = cat_all  # already includes CI + merr + MV
    return l_white, l_ci, l_merr, l_cat


def main():
    parser = argparse.ArgumentParser(description="Plot completeness per criterion step (white/CI/merr/MV).")
    parser.add_argument("--main-dir", type=str, default=".", help="Project root (default: current dir)")
    parser.add_argument("--nframe", type=int, default=4, help="Number of frames to aggregate")
    parser.add_argument("--reff", type=float, required=True, help="Reff value to use (e.g. 9)")
    args = parser.parse_args()

    main_dir = Path(args.main_dir).resolve()
    reff = float(args.reff)
    nframe = args.nframe

    mass, age, mag5 = load_physprop(main_dir, reff, nframe)
    l_white, l_ci, l_merr, l_cat = load_labels(main_dir, reff, nframe)

    if len(mass) != len(l_white):
        n = min(len(mass), len(l_white))
        mass = mass[:n]
        age = age[:n]
        mag5 = mag5[:n]
        l_white = l_white[:n]
        l_ci = l_ci[:n]
        l_merr = l_merr[:n]
        l_cat = l_cat[:n]

    labels = {
        "white": l_white,
        "white+CI": l_ci,
        "white+CI+4band": l_merr,
        "full_catalogue": l_cat,
    }
    colors = {
        "white": "C0",
        "white+CI": "C1",
        "white+CI+4band": "C2",
        "full_catalogue": "C3",
    }
    linestyles = {
        "white": "-",
        "white+CI": "--",
        "white+CI+4band": "-.",
        "full_catalogue": ":",
    }

    filter_names = ["F275W", "F336W", "F435W", "F555W", "F814W"]

    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    # vs mass
    ax = axes[0]
    for key, lab in labels.items():
        cx, cy, _ = binned_completeness_mag(np.log10(np.maximum(mass, 1e-10)), lab, n_bins=30)
        if len(cx) == 0:
            continue
        order = np.argsort(cx)
        cy = cy[order] * 100.0
        cx = cx[order]
        ax.plot(cx, cy, label=key, color=colors[key], linestyle=linestyles[key])
    ax.set_xlabel(r"$\log_{10}$(mass / M$_\odot$)")
    ax.set_ylabel("Completeness [%]")
    ax.set_title("vs mass (per criterion)")
    ax.set_ylim(-5, 105)
    ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3)

    # vs age
    ax = axes[1]
    for key, lab in labels.items():
        cx, cy, _ = binned_completeness_mag(np.log10(np.maximum(age, 1e-7)), lab, n_bins=30)
        if len(cx) == 0:
            continue
        order = np.argsort(cx)
        cy = cy[order] * 100.0
        cx = cx[order]
        ax.plot(cx, cy, label=key, color=colors[key], linestyle=linestyles[key])
    ax.set_xlabel(r"$\log_{10}$(age / yr)")
    ax.set_ylabel("Completeness [%]")
    ax.set_title("vs age (per criterion)")
    ax.set_ylim(-5, 105)
    ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3)

    # vs F555W (V-band) mag on top row right
    ax = axes[2]
    vmag = mag5[:, 3]
    for key, lab in labels.items():
        cx, cy, _ = binned_completeness_mag(vmag, lab, n_bins=30)
        if len(cx) == 0:
            continue
        order = np.argsort(cx)
        cy = cy[order] * 100.0
        cx = cx[order]
        ax.plot(cx, cy, label=key, color=colors[key], linestyle=linestyles[key])
    ax.set_xlabel("mag (F555W)")
    ax.set_ylabel("Completeness [%]")
    ax.set_title("vs F555W (per criterion)")
    ax.set_ylim(-5, 105)
    ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3)

    # Remaining 5 plots: per-band mags F275W, F336W, F435W, F555W, F814W
    band_indices = [0, 1, 2, 3, 4]
    for i, bi in enumerate(band_indices):
        ax = axes[3 + i]
        m = mag5[:, bi]
        for key, lab in labels.items():
            cx, cy, _ = binned_completeness_mag(m, lab, n_bins=30)
            if len(cx) == 0:
                continue
            order = np.argsort(cx)
            cy = cy[order] * 100.0
            cx = cx[order]
            ax.plot(cx, cy, label=key, color=colors[key], linestyle=linestyles[key])
        ax.set_xlabel(f"mag ({filter_names[bi]})")
        ax.set_ylabel("Completeness [%]")
        ax.set_title(f"vs {filter_names[bi]} (per criterion)")
        ax.set_ylim(-5, 105)
        ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for j in range(3 + len(band_indices), len(axes)):
        axes[j].set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle(f"Criterion-wise completeness ({GALAXY}, reff={reff}, {nframe} frames)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    diag_dir = ROOT / GALAXY / "white" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_path = diag_dir / f"criteria_completeness_all{nframe}frames_{nframe*500}cl_{OUTNAME}_reff{reff:.2f}.png"
    fig.savefig(out_path, dpi=140)
    print(f"Saved criterion-wise completeness to {out_path}")


if __name__ == "__main__":
    main()

