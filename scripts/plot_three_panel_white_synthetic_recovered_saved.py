#!/usr/bin/env python3
"""
Saved copy of three-panel plotting code (full frame, RA/Dec axes, galaxy FoV in black).
Three-panel figure (vertical layout):
  Top    = white (science) image
  Middle = white + inserted clusters (blue dots)
  Bottom = real image with recovered (C0) vs not recovered (C1) — colorblind-friendly.
Axes: RA (J2000) / Dec (J2000). Top and middle show galaxy field-of-view outline in black.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Plot white | white+inserted | real with recovered/non-recovered")
    parser.add_argument("--galaxy", type=str, default="ngc628-c", help="Galaxy ID")
    parser.add_argument("--outname", type=str, default="test", help="Pipeline outname")
    parser.add_argument("--reff", type=float, default=5.0, help="reff value (e.g. 5)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--filter", type=str, default="f555w", help="Real image filter for right panel (e.g. f555w)")
    parser.add_argument("--save", type=str, default=None, help="Output figure path (default: diagnostics/three_panel_...)")
    parser.add_argument("--cutout", type=int, default=0, help="Side length of cutout in pixels (0 = full frame, 100%%; default 0)")
    parser.add_argument("--cx", type=int, default=None, help="Center of cutout x (default: image center)")
    parser.add_argument("--cy", type=int, default=None, help="Center of cutout y (default: image center)")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI")
    args = parser.parse_args()

    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Need astropy and matplotlib: {e}", file=sys.stderr)
        return 1

    galaxy_id = args.galaxy
    outname = args.outname
    reff = args.reff
    frame_id = args.frame
    filt = args.filter
    cutout = args.cutout
    cx_arg = args.cx
    cy_arg = args.cy

    gal_dir = ROOT / galaxy_id
    white_dir = gal_dir / "white"
    synth_dir = white_dir / "synthetic_fits"
    matched_dir = white_dir / "matched_coords"
    cat_dir = white_dir / "catalogue"

    white_science = gal_dir / f"{galaxy_id}_white.fits"
    if not white_science.exists():
        white_science = list(gal_dir.glob("*white*.fits"))
        if not white_science:
            print(f"ERROR: No white science FITS in {gal_dir}", file=sys.stderr)
            return 1
        white_science = white_science[0]

    white_position_path = white_dir / f"white_position_{frame_id}_{outname}_reff{reff:.2f}.txt"
    matched_path = matched_dir / f"matched_frame{frame_id}_{outname}_reff{reff:.2f}.txt"
    cluster_ids_path = matched_dir / f"matched_frame{frame_id}_{outname}_reff{reff:.2f}_cluster_ids.txt"
    cat_path = cat_dir / f"catalogue_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    if not white_position_path.exists():
        print(f"ERROR: Missing white_position file: {white_position_path}", file=sys.stderr)
        return 1
    if not matched_path.exists() or not cluster_ids_path.exists() or not cat_path.exists():
        print(f"ERROR: Missing matched_coords, cluster_ids, or catalogue for frame={frame_id} reff={reff}", file=sys.stderr)
        return 1

    real_matches = list(gal_dir.glob(f"*{filt}*drc.fits")) + list(gal_dir.glob(f"*{filt.upper()}*drc.fits"))
    if not real_matches:
        print(f"ERROR: No real FITS for filter {filt} in {gal_dir}", file=sys.stderr)
        return 1
    real_path = real_matches[0]

    with fits.open(white_science) as hdul:
        img_white = np.asarray(hdul[0].data, dtype=float).squeeze()
        header_white = hdul[0].header
    ny, nx = img_white.shape
    try:
        wcs_full = WCS(header_white)
        if wcs_full.naxis > 2:
            wcs_full = wcs_full.celestial
    except Exception:
        wcs_full = None

    with fits.open(real_path) as hdul:
        img_real = np.asarray(hdul[0].data, dtype=float).squeeze()

    x0, y0, x1, y1 = 0, 0, nx, ny
    if cutout > 0 and (nx >= cutout and ny >= cutout):
        cx = cx_arg if cx_arg is not None else nx // 2
        cy = cy_arg if cy_arg is not None else ny // 2
        h = cutout // 2
        x0, x1 = max(0, cx - h), min(nx, cx + h)
        y0, y1 = max(0, cy - h), min(ny, cy + h)
        img_white = img_white[y0:y1, x0:x1]
        img_real = img_real[y0:y1, x0:x1]
        if wcs_full is not None:
            wcs_cut = wcs_full[y0:y1, x0:x1]
        else:
            wcs_cut = None
    else:
        wcs_cut = wcs_full

    if wcs_full is not None:
        pix_corners_x = np.array([1, nx, nx, 1, 1], dtype=float)
        pix_corners_y = np.array([1, 1, ny, ny, 1], dtype=float)
        ra_corners, dec_corners = wcs_full.all_pix2world(pix_corners_x, pix_corners_y, 0)
    else:
        ra_corners = dec_corners = None

    wp = np.loadtxt(white_position_path)
    if wp.ndim == 1:
        wp = wp.reshape(1, -1)
    x_inj = wp[:, 0]
    y_inj = wp[:, 1]
    if cutout > 0 and (nx >= cutout and ny >= cutout):
        inside_inj = (x_inj >= x0) & (x_inj < x1) & (y_inj >= y0) & (y_inj < y1)
        x_inj_plot = x_inj[inside_inj] - x0
        y_inj_plot = y_inj[inside_inj] - y0
    else:
        x_inj_plot = x_inj
        y_inj_plot = y_inj

    coords = np.loadtxt(matched_path)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    x_matched = coords[:, 0]
    y_matched = coords[:, 1]
    cluster_ids = np.loadtxt(cluster_ids_path, dtype=np.intp)
    if cluster_ids.ndim == 0:
        cluster_ids = np.array([cluster_ids])

    cat = __import__("pandas").read_parquet(cat_path)
    cat_sub = cat[(cat["frame_id"] == frame_id) & (np.abs(cat["reff"] - reff) < 0.01)]
    in_cat = dict(zip(cat_sub["cluster_id"], cat_sub["in_catalogue"]))

    recovered = np.array([in_cat.get(cid, 0) == 1 for cid in cluster_ids])
    x_0based = x_matched - 1.0
    y_0based = y_matched - 1.0
    if cutout > 0:
        inside = (
            (x_0based >= x0) & (x_0based < x1) &
            (y_0based >= y0) & (y_0based < y1)
        )
        x_plot = x_0based[inside] - x0
        y_plot = y_0based[inside] - y0
        rec_plot = recovered[inside]
        nrec_plot = ~rec_plot
    else:
        x_plot = x_0based
        y_plot = y_0based
        rec_plot = recovered
        nrec_plot = ~recovered

    if wcs_cut is not None:
        fig = plt.figure(figsize=(5, 12))
        axes = [
            fig.add_subplot(3, 1, 1, projection=wcs_cut),
            fig.add_subplot(3, 1, 2, projection=wcs_cut),
            fig.add_subplot(3, 1, 3, projection=wcs_cut),
        ]
        for ax in axes:
            try:
                ax.coords[0].set_axislabel("RA (J2000)", fontsize=8)
                ax.coords[1].set_axislabel("Dec (J2000)", fontsize=8)
                ax.coords[0].set_ticklabel(fontsize=7)
                ax.coords[1].set_ticklabel(fontsize=7)
            except (KeyError, IndexError):
                ax.set_xlabel("RA (J2000)", fontsize=8)
                ax.set_ylabel("Dec (J2000)", fontsize=8)
                ax.tick_params(labelsize=7)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(5, 12))
        for ax in axes:
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

    def show_im(ax, data, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.nanpercentile(data, 2)
        if vmax is None:
            vmax = np.nanpercentile(data, 98)
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    show_im(axes[0], img_white)
    if ra_corners is not None and dec_corners is not None:
        axes[0].plot(ra_corners, dec_corners, "k-", lw=1)
    axes[0].set_title("White", fontsize=9)

    show_im(axes[1], img_white)
    if ra_corners is not None and dec_corners is not None:
        axes[1].plot(ra_corners, dec_corners, "k-", lw=1)
    if len(x_inj_plot) > 0:
        axes[1].scatter(x_inj_plot, y_inj_plot, c="blue", s=5, alpha=0.85, edgecolors="none")
    axes[1].set_title("White + injected", fontsize=9)

    show_im(axes[2], img_real)
    if len(x_plot) > 0:
        if np.any(rec_plot):
            axes[2].scatter(x_plot[rec_plot], y_plot[rec_plot], c="C0", s=6, alpha=0.9, label="Recovered", edgecolors="none")
        if np.any(nrec_plot):
            axes[2].scatter(x_plot[nrec_plot], y_plot[nrec_plot], c="C1", s=6, alpha=0.9, label="Not recovered", edgecolors="none")
        axes[2].legend(loc="upper right", fontsize=7)
    axes[2].set_title("Real: recovered / not recovered", fontsize=9)

    fig.suptitle(f"{galaxy_id}  reff={reff}  ({np.sum(recovered)}/{len(recovered)} recovered)", fontsize=9)
    plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)

    if args.save:
        out_path = Path(args.save)
    else:
        diag_dir = white_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_path = diag_dir / f"three_panel_white_synthetic_recovered_{outname}_frame{frame_id}_reff{reff:.2f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
