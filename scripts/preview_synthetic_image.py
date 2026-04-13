#!/usr/bin/env python3
"""
Quick preview of synthetic (inserted) images: white and one filter (F555W).
Reads one frame FITS + coord file, shows a cutout and optional overlay of injected positions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
GAL = "ngc628-c"


def load_coords(coord_path):
    """Load x, y from coord file (format: x y mag). Assume 1-based pixel coords -> convert to 0-based index."""
    data = np.loadtxt(coord_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x, y = data[:, 0], data[:, 1]
    # Common convention: FITS/IRAF use 1-based; convert to 0-based for numpy
    x0 = np.asarray(x, dtype=float) - 1
    y0 = np.asarray(y, dtype=float) - 1
    return x0, y0


def main():
    # White synthetic frame (one frame)
    white_fits = ROOT / GAL / "white" / "synthetic_fits" / "ngc628-c_WFC3_UVISfF336W_frame0_test_reff9.00.fits"
    white_coord = ROOT / GAL / "white" / "synthetic_fits" / "ngc628-c_WFC3_UVISfF336W_reff9.00pc_sources_frame0_test.txt"
    # F555W synthetic frame
    f555w_fits = ROOT / GAL / "f555w" / "synthetic_fits" / "ngc628-c_acsff555w_frame0_test_reff9.00.fits"

    if not white_fits.exists():
        print(f"Not found: {white_fits}")
        return
    if not white_coord.exists():
        print(f"Not found: {white_coord}")
        return

    # Load white image (first science extension if MEF)
    with fits.open(white_fits) as hdul:
        for ext in hdul:
            if hasattr(ext, "data") and ext.data is not None and ext.data.ndim >= 2:
                img_white = np.asarray(ext.data, dtype=float)
                break
        else:
            img_white = hdul[0].data.astype(float)

    xc, yc = load_coords(white_coord)

    # Cutout: pick a region that has several clusters (e.g. center-ish)
    h, w = img_white.shape
    # Center cutout 800 x 800
    size = 800
    cx, cy = w // 2, h // 2
    x0, x1 = max(0, cx - size // 2), min(w, cx + size // 2)
    y0, y1 = max(0, cy - size // 2), min(h, cy + size // 2)
    cut_white = img_white[y0:y1, x0:x1].copy()
    cut_white[cut_white <= 0] = np.nanmin(cut_white[cut_white > 0]) if np.any(cut_white > 0) else 1e-10

    # Mask coords to this cutout (use 0-based indices)
    in_cut = (xc >= x0) & (xc < x1) & (yc >= y0) & (yc < y1)
    xc_cut = xc[in_cut] - x0
    yc_cut = yc[in_cut] - y0

    # F555W cutout (same region) if exists
    if f555w_fits.exists():
        with fits.open(f555w_fits) as hdul:
            for ext in hdul:
                if hasattr(ext, "data") and ext.data is not None and ext.data.ndim >= 2:
                    img_f555w = np.asarray(ext.data, dtype=float)
                    break
            else:
                img_f555w = hdul[0].data.astype(float)
        cut_f555w = img_f555w[y0:y1, x0:x1].copy()
        cut_f555w[cut_f555w <= 0] = np.nanmin(cut_f555w[cut_f555w > 0]) if np.any(cut_f555w > 0) else 1e-10
        two_panel = True
    else:
        two_panel = False

    # Plot
    fig, axes = plt.subplots(1, 2 if two_panel else 1, figsize=(12 if two_panel else 7, 6))
    if not two_panel:
        axes = [axes]
    vmin = np.nanpercentile(cut_white, 1)
    vmax = np.nanpercentile(cut_white, 99)
    axes[0].imshow(cut_white, origin="lower", cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[0].scatter(xc_cut, yc_cut, s=8, c="lime", alpha=0.9, edgecolors="none")
    axes[0].set_title("White (frame0, reff=9) – inserted positions")
    axes[0].set_xlabel("x (cutout)")
    axes[0].set_ylabel("y (cutout)")

    if two_panel:
        vmin2 = np.nanpercentile(cut_f555w, 1)
        vmax2 = np.nanpercentile(cut_f555w, 99)
        axes[1].imshow(cut_f555w, origin="lower", cmap="gray", norm=LogNorm(vmin=vmin2, vmax=vmax2))
        axes[1].scatter(xc_cut, yc_cut, s=8, c="red", alpha=0.7, edgecolors="none")
        axes[1].set_title("F555W (frame0, reff=9) – same positions")
        axes[1].set_xlabel("x (cutout)")

    plt.tight_layout()
    out = ROOT / GAL / "white" / "diagnostics" / "preview_inserted_image_frame0_reff9.00.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
