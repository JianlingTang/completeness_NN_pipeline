#!/usr/bin/env python3
"""
Three-panel figure (vertical layout):
  Top / Middle = white science WCS (wcs_full); optional --cutout shows a chip window in sky coords.
  Middle       = synthetic + injected positions (blue circles).
  Bottom       = zoom on full chip: --use-fk4-bottom (DS9 FK4 box) or pixel box / ICRS / FK4 rect.

Top and middle share the same projection as the white FITS header; bottom uses a WCS slice of that same WCS.
"""

import argparse
import sys
from pathlib import Path

from astropy.visualization.wcsaxes import Quadrangle
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import argparse
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.patches import Rectangle
mpl.rcParams.update({
    "figure.figsize": (5., 5.0),
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.major.size": 6,
    "ytick.minor.size": 3,
    "text.usetex": False,  # True if you use LaTeX backend
})

plt.rc("text", usetex=False)
plt.rc("font", family="serif")
plt.rc("mathtext", fontset="cm")   # Computer Modern
from astropy.coordinates import SkyCoord

try:
    from regions import PolygonSkyRegion
    HAS_REGIONS = True
except ImportError:
    PolygonSkyRegion = None
    HAS_REGIONS = False

ra_deg = [
    23.5095682,
    23.4682076,
    23.4554267,
    23.4664910,
    23.4775508,
    23.5067723,
    23.5297127,
    23.5297100,
]

dec_deg = [
    15.5604814,
    15.5418337,
    15.5360402,
    15.5103735,
    15.4831385,
    15.4970240,
    15.5084584,
    15.5084922,
]

vertices = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="fk4")
region_sky = PolygonSkyRegion(vertices=vertices) if HAS_REGIONS else None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Plot white | white+inserted | zoomed recovered/non-recovered"
    )
    parser.add_argument("--galaxy", type=str, default="ngc628-c", help="Galaxy ID")
    parser.add_argument("--outname", type=str, default="test", help="Pipeline outname")
    parser.add_argument("--reff", type=float, default=5.0, help="reff value (e.g. 5)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--filter", type=str, default="f555w", help="Kept for compatibility")
    parser.add_argument("--save", type=str, default=None, help="Output figure path")
    parser.add_argument("--cutout", type=int, default=0, help="Side length of cutout in pixels (0 = full frame)")
    parser.add_argument("--cx", type=int, default=None, help="Center of cutout x (default: image center)")
    parser.add_argument("--cy", type=int, default=None, help="Center of cutout y (default: image center)")
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI")

    # Bottom panel zoom region. Option A: ICRS deg (center + size + angle)
    parser.add_argument("--region-ra-center", type=float, default=None, help="Bottom panel zoom: center RA (ICRS deg)")
    parser.add_argument("--region-dec-center", type=float, default=None, help="Bottom panel zoom: center Dec (ICRS deg)")
    parser.add_argument("--region-width", type=float, default=None, help="Bottom panel zoom: width (deg)")
    parser.add_argument("--region-height", type=float, default=None, help="Bottom panel zoom: height (deg)")
    parser.add_argument("--region-angle", type=float, default=0.0, help="Bottom panel zoom: angle (deg)")

    # FK4 box
    parser.add_argument("--use-fk4-bottom", action="store_true", help="Use DS9 FK4 box for bottom panel")
    parser.add_argument("--region-fk4-ra", type=str, default=None, help="FK4 center RA sexagesimal")
    parser.add_argument("--region-fk4-dec", type=str, default=None, help="FK4 center Dec sexagesimal")
    parser.add_argument("--region-fk4-width", type=float, default=None, help="Bottom width in deg (FK4 box)")
    parser.add_argument("--region-fk4-height", type=float, default=None, help="Bottom height in deg (FK4 box)")
    parser.add_argument("--region-fk4-angle", type=float, default=None, help="Rotation deg (DS9 angle)")

    # FK4 rect
    parser.add_argument("--region-ra1", type=float, default=None, help="Bottom panel zoom: RA min (FK4 deg)")
    parser.add_argument("--region-ra2", type=float, default=None, help="Bottom panel zoom: RA max (FK4 deg)")
    parser.add_argument("--region-dec1", type=float, default=None, help="Bottom panel zoom: Dec min (FK4 deg)")
    parser.add_argument("--region-dec2", type=float, default=None, help="Bottom panel zoom: Dec max (FK4 deg)")

    # Pixel zoom
    parser.add_argument("--bot-cx", type=float, default=1763.9872, help="Bottom zoom center x")
    parser.add_argument("--bot-cy", type=float, default=2030.9141, help="Bottom zoom center y")
    parser.add_argument(
        "--bot-coords-middle",
        action="store_true",
        help="Interpret --bot-cx/--bot-cy as pixels in the middle-panel displayed image",
    )
    parser.add_argument("--bot-width", type=float, default=1117.9524, help="Bottom region width (pixels)")
    parser.add_argument("--bot-height", type=float, default=587.64165, help="Bottom region height (pixels)")
    parser.add_argument(
        "--bot-corner-quarter",
        action="store_true",
        help="Bottom: use [0,nx/4)×[0,ny/4) instead of center+size pixels",
    )

    args = parser.parse_args()

    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import pandas as pd
    except ImportError as e:
        print(f"Need astropy, matplotlib, pandas: {e}", file=sys.stderr)
        return 1

    def show_im(ax, data, cmap="gray_r", vmin=None, vmax=None, transform=None, extent_xy=None, mask_invalid=False):
        arr = np.asarray(data, dtype=float).copy()

        if mask_invalid:
            invalid = ~np.isfinite(arr) | (arr == 0)
            arr = np.ma.array(arr, mask=invalid)

        finite_vals = np.asarray(data, dtype=float)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size == 0:
            finite_vals = np.array([0.0, 1.0])

        if vmin is None:
            vmin = np.nanpercentile(finite_vals, 2)
        if vmax is None:
            vmax = np.nanpercentile(finite_vals, 98)

        cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)

        if transform is not None:
            if extent_xy is not None:
                xl, xr, yb, yt = extent_xy
            else:
                ny_, nx_ = arr.shape
                xl, xr, yb, yt = 0.0, float(nx_), 0.0, float(ny_)
            return ax.imshow(
                arr,
                origin="lower",
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
                transform=transform,
                extent=(xl, xr, yb, yt),
            )

        return ax.imshow(
            arr,
            origin="lower",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )

    def compute_valid_bbox(img, threshold=None):
        data = np.asarray(img, dtype=float)
        valid = np.isfinite(data)
        if threshold is None:
            valid &= (data != 0)
        else:
            valid &= (data > threshold)

        if not np.any(valid):
            return None

        yy, xx = np.where(valid)
        x_min = xx.min()
        x_max = xx.max()
        y_min = yy.min()
        y_max = yy.max()
        return x_min, x_max, y_min, y_max

    def compute_world_box_limits_from_center(ra_center, dec_center, width_deg, height_deg, angle_deg=0.0):
        rad = np.deg2rad(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        half_w = width_deg / 2.0
        half_h = height_deg / 2.0
        dx = np.array([half_w, -half_w, -half_w, half_w])
        dy = np.array([half_h, half_h, -half_h, -half_h])
        dx_rot = dx * c - dy * s
        dy_rot = dx * s + dy * c
        cd = np.cos(np.deg2rad(dec_center))
        cd = cd if abs(cd) > 1e-10 else 1.0
        ra_corners = ra_center + dx_rot / cd
        dec_corners = dec_center + dy_rot
        return (np.min(ra_corners), np.max(ra_corners)), (np.min(dec_corners), np.max(dec_corners))

    galaxy_id = args.galaxy
    outname = args.outname
    reff = args.reff
    frame_id = args.frame
    cutout = args.cutout

    gal_dir = ROOT / galaxy_id
    white_dir = gal_dir / "white"
    synth_dir = white_dir / "synthetic_fits"
    matched_dir = white_dir / "matched_coords"
    cat_dir = white_dir / "catalogue"

    white_science = gal_dir / f"{galaxy_id}_white.fits"
    if not white_science.exists():
        white_candidates = list(gal_dir.glob("*white*.fits"))
        if not white_candidates:
            print(f"ERROR: No white science FITS in {gal_dir}", file=sys.stderr)
            return 1
        white_science = white_candidates[0]

    white_position_path = white_dir / f"white_position_{frame_id}_{outname}_reff{reff:.2f}.txt"
    matched_path = matched_dir / f"matched_frame{frame_id}_{outname}_reff{reff:.2f}.txt"
    cluster_ids_path = matched_dir / f"matched_frame{frame_id}_{outname}_reff{reff:.2f}_cluster_ids.txt"
    cat_path = cat_dir / f"catalogue_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"

    if not white_position_path.exists():
        print(f"ERROR: Missing white_position file: {white_position_path}", file=sys.stderr)
        return 1
    if not matched_path.exists() or not cluster_ids_path.exists() or not cat_path.exists():
        print(
            f"ERROR: Missing matched_coords, cluster_ids, or catalogue for frame={frame_id} reff={reff}",
            file=sys.stderr,
        )
        return 1

    with fits.open(white_science) as hdul:
        img_white_full = np.asarray(hdul[0].data, dtype=float).squeeze()
        header_white = hdul[0].header

    if img_white_full.ndim != 2:
        print(f"ERROR: Expected 2D white image, got shape {img_white_full.shape}", file=sys.stderr)
        return 1

    ny_full, nx_full = img_white_full.shape

    try:
        wcs_full = WCS(header_white)
        if wcs_full.naxis > 2:
            wcs_full = wcs_full.celestial
    except Exception:
        wcs_full = None

    synth_matches = list(synth_dir.glob(f"*frame{frame_id}_{outname}_reff{reff:.2f}.fits"))
    if not synth_matches:
        print(
            f"ERROR: No synthetic white FITS *frame{frame_id}_{outname}_reff{reff:.2f}.fits in {synth_dir}",
            file=sys.stderr,
        )
        return 1

    synthetic_white_path = synth_matches[0]
    with fits.open(synthetic_white_path) as hdul:
        img_synthetic_full = np.asarray(hdul[0].data, dtype=float).squeeze()

    if img_synthetic_full.shape != (ny_full, nx_full):
        print(
            f"WARNING: Synthetic white shape {img_synthetic_full.shape} != white ({ny_full}, {nx_full})",
            file=sys.stderr,
        )
        if img_synthetic_full.shape[0] >= ny_full and img_synthetic_full.shape[1] >= nx_full:
            img_synthetic_full = img_synthetic_full[:ny_full, :nx_full]
        else:
            print("ERROR: Synthetic image is smaller than white image.", file=sys.stderr)
            return 1

    x0, y0, x1, y1 = 0, 0, nx_full, ny_full
    if cutout > 0 and nx_full >= cutout and ny_full >= cutout:
        cx = args.cx if args.cx is not None else nx_full // 2
        cy = args.cy if args.cy is not None else ny_full // 2
        h = cutout // 2
        x0, x1 = max(0, cx - h), min(nx_full, cx + h)
        y0, y1 = max(0, cy - h), min(ny_full, cy + h)

    img_white = img_white_full[y0:y1, x0:x1]
    img_synthetic = img_synthetic_full[y0:y1, x0:x1]

    wp = np.loadtxt(white_position_path)
    if wp.ndim == 1:
        wp = wp.reshape(1, -1)

    x_col = np.asarray(wp[:, 0], dtype=float)
    y_row = np.asarray(wp[:, 1], dtype=float)
    n_inj = len(x_col)

    cat = pd.read_parquet(cat_path)
    cat_sub = cat[(cat["frame_id"] == frame_id) & (np.abs(cat["reff"] - reff) < 0.01)]
    in_cat = dict(zip(cat_sub["cluster_id"], cat_sub["in_catalogue"]))
    inj_recovered = np.array([in_cat.get(i, 0) == 1 for i in range(n_inj)], dtype=bool)

    inside_inj = (x_col >= x0) & (x_col < x1) & (y_row >= y0) & (y_row < y1)
    x_col_plot = x_col[inside_inj]
    y_row_plot = y_row[inside_inj]

    coords = np.loadtxt(matched_path)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    x_matched = np.asarray(coords[:, 0], dtype=float)
    y_matched = np.asarray(coords[:, 1], dtype=float)

    cluster_ids = np.loadtxt(cluster_ids_path, dtype=np.intp)
    if cluster_ids.ndim == 0:
        cluster_ids = np.array([cluster_ids])

    recovered = np.array([in_cat.get(cid, 0) == 1 for cid in cluster_ids], dtype=bool)

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
    else:
        x_plot = x_0based
        y_plot = y_0based
        rec_plot = recovered

    ra_lim = dec_lim = None

    if all(v is not None for v in (args.region_ra_center, args.region_dec_center, args.region_width, args.region_height)) and wcs_full is not None:
        ra_lim, dec_lim = compute_world_box_limits_from_center(
            args.region_ra_center,
            args.region_dec_center,
            args.region_width,
            args.region_height,
            args.region_angle or 0.0,
        )

    if ra_lim is None and dec_lim is None and wcs_full is not None:
        fk4_use = args.use_fk4_bottom or (
            args.region_fk4_ra is not None and
            args.region_fk4_dec is not None and
            args.region_fk4_width is not None and
            args.region_fk4_height is not None
        )

        if fk4_use:
            if args.use_fk4_bottom:
                ra_fs = args.region_fk4_ra or "1h34m01.826s"
                dec_fs = args.region_fk4_dec or "15d30m45.38s"
                fw = args.region_fk4_width if args.region_fk4_width is not None else 0.0153037
                fh = args.region_fk4_height if args.region_fk4_height is not None else 0.0153037
                fang = args.region_fk4_angle if args.region_fk4_angle is not None else 359.88329
            else:
                ra_fs = args.region_fk4_ra
                dec_fs = args.region_fk4_dec
                fw = args.region_fk4_width
                fh = args.region_fk4_height
                fang = args.region_fk4_angle if args.region_fk4_angle is not None else 0.0

            c_icrs = SkyCoord(ra_fs, dec_fs, frame="fk4").icrs
            ra_lim, dec_lim = compute_world_box_limits_from_center(
                float(c_icrs.ra.deg),
                float(c_icrs.dec.deg),
                fw,
                fh,
                fang,
            )

    if ra_lim is None and dec_lim is None and wcs_full is not None:
        if all(v is not None for v in (args.region_ra1, args.region_ra2, args.region_dec1, args.region_dec2)):
            ra_lo, ra_hi = min(args.region_ra1, args.region_ra2), max(args.region_ra1, args.region_ra2)
            dec_lo, dec_hi = min(args.region_dec1, args.region_dec2), max(args.region_dec1, args.region_dec2)

            corners_fk4 = SkyCoord(
                [ra_lo, ra_hi, ra_lo, ra_hi],
                [dec_lo, dec_lo, dec_hi, dec_hi],
                unit="deg",
                frame="fk4",
            )
            corners_icrs = corners_fk4.transform_to("icrs")
            ra_lim = (corners_icrs.ra.deg.min(), corners_icrs.ra.deg.max())
            dec_lim = (corners_icrs.dec.deg.min(), corners_icrs.dec.deg.max())

    if args.bot_corner_quarter:
        bot_px0, bot_px1 = 0, max(1, nx_full // 4)
        bot_py0, bot_py1 = 0, max(1, ny_full // 4)
        bot_px1 = min(nx_full, bot_px1)
        bot_py1 = min(ny_full, bot_py1)
    elif ra_lim is not None and dec_lim is not None and wcs_full is not None:
        ras = np.array([ra_lim[0], ra_lim[1], ra_lim[0], ra_lim[1]], dtype=float)
        decs = np.array([dec_lim[0], dec_lim[0], dec_lim[1], dec_lim[1]], dtype=float)
        xp, yp = wcs_full.all_world2pix(ras, decs, 0)
        bot_px0 = int(np.clip(np.floor(np.min(xp)), 0, max(0, nx_full - 1)))
        bot_px1 = int(np.clip(np.ceil(np.max(xp)), 1, nx_full))
        bot_py0 = int(np.clip(np.floor(np.min(yp)), 0, max(0, ny_full - 1)))
        bot_py1 = int(np.clip(np.ceil(np.max(yp)), 1, ny_full))
        if bot_px1 <= bot_px0:
            bot_px0, bot_px1 = 0, nx_full
        if bot_py1 <= bot_py0:
            bot_py0, bot_py1 = 0, ny_full
    else:
        if args.bot_coords_middle:
            cx_bot = float(args.bot_cx) + float(x0)
            cy_bot = float(args.bot_cy) + float(y0)
        else:
            cx_bot = float(args.bot_cx)
            cy_bot = float(args.bot_cy)

        half_w = args.bot_width / 2.0
        half_h = args.bot_height / 2.0
        bot_px0 = int(np.floor(cx_bot - half_w))
        bot_px1 = int(np.ceil(cx_bot + half_w))
        bot_py0 = int(np.floor(cy_bot - half_h))
        bot_py1 = int(np.ceil(cy_bot + half_h))

        bot_px0 = max(0, bot_px0)
        bot_py0 = max(0, bot_py0)
        bot_px1 = min(nx_full, max(bot_px0 + 1, bot_px1))
        bot_py1 = min(ny_full, max(bot_py0 + 1, bot_py1))

    img_bot = np.asarray(img_synthetic_full[bot_py0:bot_py1, bot_px0:bot_px1], dtype=float)
    wcs_bot = wcs_full[bot_py0:bot_py1, bot_px0:bot_px1] if wcs_full is not None else None
    bottom_region_sky = None
    if HAS_REGIONS and wcs_full is not None:
        x_corners = np.array([bot_px0, bot_px1, bot_px1, bot_px0], dtype=float)
        y_corners = np.array([bot_py0, bot_py0, bot_py1, bot_py1], dtype=float)

        sky_corners = wcs_full.pixel_to_world(x_corners, y_corners)
        bottom_region_sky = PolygonSkyRegion(vertices=sky_corners)

    finite_syn = img_synthetic_full[np.isfinite(img_synthetic_full)]
    if finite_syn.size == 0:
        print("ERROR: synthetic image has no finite values.", file=sys.stderr)
        return 1

    vmin_global = np.nanpercentile(finite_syn, 2)
    vmax_global = np.nanpercentile(finite_syn, 98)

    fig = plt.figure(figsize=(3.5, 7.5))
    fig.supylabel("Dec (J2000)", fontsize=11, x=0.01)
    gs = GridSpec(
        3, 1,
        height_ratios=[1, 1, 1],
        hspace=0.1,
        figure=fig,
    )

    pixel_region = (
        region_sky.to_pixel(wcs_full) if (HAS_REGIONS and region_sky is not None and wcs_full is not None) else None
    )

    if wcs_full is not None:
        ax0 = fig.add_subplot(gs[0, 0], projection=wcs_full)
        ax1 = fig.add_subplot(gs[1, 0], projection=wcs_full)
        ax2 = fig.add_subplot(gs[2, 0], projection=wcs_bot)
    else:
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0])

    axes = [ax0, ax1, ax2]
    # keep panel boxes aligned; do not force square boxes for all WCS axes
    ax0.set_anchor("N")
    ax1.set_anchor("N")
    ax2.set_anchor("N")

    # only enforce 1:1 data aspect on the bottom zoom panel
    ax2.set_aspect("equal", adjustable="box")

    if wcs_full is not None:
        for i, ax in enumerate(axes):
            try:
                ax.coords[0].set_ticklabel()
                ax.coords[1].set_ticklabel()
                ax.coords[1].set_axislabel("")

                if i == 0:
                    ax.coords[0].set_axislabel("")
                    ax.coords[0].set_ticklabel_visible(False)
                elif i == 1:
                    ax.coords[0].set_axislabel("")
                    ax.coords[0].set_ticklabel_visible(True)
                else:
                    ax.coords[0].set_axislabel("RA (J2000)", fontsize=10)
                    ax.coords[0].set_ticklabel_visible(True)
            except (KeyError, IndexError, AttributeError):
                ax.set_ylabel("", fontsize=10)
                if i == 0 or i == 1:
                    ax.set_xlabel("")
                    ax.set_xticklabels([], fontsize=10)
                else:
                    ax.set_xlabel("RA (J2000)")
                    ax.tick_params(axis="x", labelbottom=True, labelsize=10)
                ax.tick_params(labelsize=10)
    else:
        for i, ax in enumerate(axes):
            ax.set_aspect("equal")
            ax.tick_params()
            ax.set_ylabel("y", fontsize=10)
            if i == 0:
                ax.set_xlabel("")
                ax.set_xticklabels([], fontsize=10)
            else:
                ax.set_xlabel("x", fontsize=10)

    t0 = None
    if wcs_full is not None and hasattr(ax0, "get_transform"):
        try:
            t0 = ax0.get_transform("pixel")
        except Exception:
            t0 = None

    if t0 is not None:
        im0 = show_im(
            ax0,
            img_white,
            vmin=vmin_global,
            vmax=vmax_global,
            transform=t0,
            extent_xy=(float(x0), float(x1), float(y0), float(y1)),
            mask_invalid=True,
        )
    else:
        im0 = show_im(ax0, img_white, vmin=vmin_global, vmax=vmax_global, mask_invalid=True)

    t1 = None
    if wcs_full is not None and hasattr(ax1, "get_transform"):
        try:
            t1 = ax1.get_transform("pixel")
        except Exception:
            t1 = None

    if t1 is not None:
        im1 = show_im(
            ax1,
            img_synthetic,
            vmin=vmin_global,
            vmax=vmax_global,
            transform=t1,
            extent_xy=(float(x0), float(x1), float(y0), float(y1)),
            mask_invalid=True,
        )
    else:
        im1 = show_im(ax1, img_synthetic, vmin=vmin_global, vmax=vmax_global, mask_invalid=True)

    lw_circle = 0.6
    lw_circle_bot = 1.2
    s_circle_mid = 4
    s_circle_bot = 55

    if len(x_col_plot) > 0:
        if t1 is not None:
            ax1.scatter(
                x_col_plot,
                y_row_plot,
                s=s_circle_mid,
                alpha=1.0,
                facecolors="none",
                edgecolors="blue",
                linewidths=lw_circle,
                transform=t1,
                zorder=30,
                label="Injected",
            )
        else:
            ax1.scatter(
                x_col_plot,
                y_row_plot,
                s=s_circle_mid,
                alpha=0.8,
                facecolors="none",
                edgecolors="blue",
                linewidths=lw_circle,
                zorder=30,
                label="Injected",
            )
        if bottom_region_sky is not None:
            bottom_region_pix = bottom_region_sky.to_pixel(wcs_full)
            bottom_region_pix.plot(ax=ax1, color="black", lw=1.0, alpha=1.0, zorder=120)

    xc_all = np.asarray(x_col, dtype=float)
    yr_all = np.asarray(y_row, dtype=float)
    rec_all = np.asarray(inj_recovered, dtype=bool)

    nx_b = bot_px1 - bot_px0
    ny_b = bot_py1 - bot_py0

    in_view = (
        (xc_all >= bot_px0) & (xc_all < bot_px1) &
        (yr_all >= bot_py0) & (yr_all < bot_py1)
    )
    x_p = xc_all[in_view] - float(bot_px0)
    y_p = yr_all[in_view] - float(bot_py0)
    rec_v = rec_all[in_view]
    nrec_v = ~rec_v
    
    if pixel_region is not None:
        pixel_region.plot(ax=ax0, color="black", lw=3, alpha=1, zorder=100)
        ax0.plot([], [], color="black", lw=3, label="Field of View")

    leg0 = ax0.legend(
        loc="upper right",
        framealpha=1.0,
        facecolor="white",
        edgecolor="black"
    )
    leg0.set_zorder(1000)

    # Plot the region on the first two axes
    if pixel_region is not None:
        pixel_region.plot(ax=ax1, color="black", lw=3, alpha=1, zorder=100)
    leg1 = ax1.legend(
    loc="upper right",
    framealpha=1.0,
    facecolor="white",
    edgecolor="black"
    )
    leg1.set_zorder(1000)

    legend_obj = None

    tb = None
    if wcs_bot is not None and hasattr(ax2, "get_transform"):
        try:
            tb = ax2.get_transform("pixel")
        except Exception:
            tb = None

    if tb is not None:
        im2 = ax2.imshow(
            img_bot,
            origin="lower",
            cmap="gray_r",
            vmin=vmin_global,
            vmax=vmax_global,
            transform=tb,
            extent=(0, nx_b, 0, ny_b),
        )

        if x_p.size > 0:
            if np.any(nrec_v):
                ax2.scatter(
                    x_p[nrec_v],
                    y_p[nrec_v],
                    s=s_circle_bot,
                    alpha=1.0,
                    facecolors="none",
                    edgecolors="limegreen",
                    linewidths=lw_circle_bot,
                    label="Not recovered",
                    ls="dashed",
                    transform=tb,
                    zorder=30,
                )
            if np.any(rec_v):
                ax2.scatter(
                    x_p[rec_v],
                    y_p[rec_v],
                    s=s_circle_bot,
                    alpha=1.0,
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=lw_circle_bot,
                    label="Recovered",
                    transform=tb,
                    zorder=30,
                )

            legend_obj = ax2.legend(
                loc="upper right",
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            legend_obj.set_zorder(1000)

        try:
            ax2.coords[0].set_axislabel("RA (J2000)")
            ax2.coords[0].set_ticklabel_visible(True)
            ax2.coords[1].set_axislabel("")
        except (AttributeError, KeyError, IndexError):
            pass
    else:
        im2 = ax2.imshow(
            img_bot,
            origin="lower",
            cmap="gray_r",
            vmin=vmin_global,
            vmax=vmax_global,
        )
        ax2.set_aspect("equal")

        if x_p.size > 0:
            if np.any(nrec_v):
                ax2.scatter(
                    x_p[nrec_v],
                    y_p[nrec_v],
                    s=s_circle_bot,
                    alpha=1.0,
                    facecolors="none",
                    edgecolors="blue",
                    linewidths=lw_circle,
                    label="Not recovered",
                    zorder=30,
                )
            if np.any(rec_v):
                ax2.scatter(
                    x_p[rec_v],
                    y_p[rec_v],
                    s=s_circle_bot,
                    alpha=1.0,
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=lw_circle,
                    label="Recovered",
                    zorder=30,
                )

            legend_obj = ax2.legend(
                loc="upper left",
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            legend_obj.set_zorder(1000)

    ax2.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.99,
    bottom=0.05,
    hspace=0.02,)

    if args.save:
        out_path = Path(args.save)
    else:
        diag_dir = white_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_path = diag_dir / (
            f"three_panel_white_synthetic_recovered_{outname}_frame{frame_id}_reff{reff:.2f}.pdf"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())