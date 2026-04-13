#!/usr/bin/env python3
"""
Plot three panels: (1) initial white-light image (BAOlab input), (2) synthetic white-light
image with fake clusters added, (3) same synthetic image with injected clusters marked as
recovered (green) vs non-recovered (red) through the full pipeline (photometry + CI when
catalogue exists; otherwise white-light detection match only).

Injected positions come from generate_white_clusters: they are placed only where both the
original image and the image convolved with a Gaussian of sigma = 120 pc (in pixel scale)
have positive flux (i.e. within the galaxy/detection footprint).

Usage:
    python scripts/plot_white_injection_diagnostics.py --main-dir /path/to/project \\
        --galaxy ngc628-c --frame 0 --reff 3 --outname test

    # Optional: specify initial white FITS (otherwise discovered from main_dir / COMP_SCIFRAME)
    python scripts/plot_white_injection_diagnostics.py --main-dir /path --galaxy ngc628-c \\
        --frame 0 --reff 3 --outname test --initial-white /path/to/ngc628-c_white.fits \\
        --output diagnostic_frame0_reff3.pdf
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Field of view polygon: FK4, unit degree (DS9 format); converted to J2000 for plotting
FOV_FK4_DEG = [
    (23.5096435, 15.5604094),
    (23.4554852, 15.5360424),
    (23.4775486, 15.4832707),
    (23.5043912, 15.4960147),
    (23.5198950, 15.5035797),
    (23.5296906, 15.5084187),
    (23.5200876, 15.5338140),
]


def _fov_polygon_j2000():
    """Return FOV polygon (ra, dec) in J2000/ICRS degrees for plotting."""
    ra = np.array([p[0] for p in FOV_FK4_DEG]) * u.deg
    dec = np.array([p[1] for p in FOV_FK4_DEG]) * u.deg
    c = SkyCoord(ra=ra, dec=dec, frame="fk4")
    c_j2000 = c.transform_to("icrs")
    return np.array(c_j2000.ra.deg), np.array(c_j2000.dec.deg)

def _path_env(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).resolve() if raw else default.resolve()


def find_initial_white(main_dir: Path, galaxy_id: str, fits_path: Path | None) -> Path | None:
    """Discover initial white-light science frame (BAOlab input)."""
    gal_short = galaxy_id.split("_")[0]
    candidates = [
        fits_path / galaxy_id / "ngc628-c_white.fits",
        fits_path / galaxy_id / f"{gal_short}_white.fits",
        main_dir / galaxy_id / "ngc628-c_white.fits",
        fits_path / f"{gal_short}_white-R17v100",
    ]
    for d in candidates:
        if d.suffix == ".fits" and d.exists():
            return d
        if d.is_dir():
            for name in ("ngc628-c_white.fits", f"{gal_short}_white.fits", "white_dualpop_s2n_white_remake.fits"):
                p = d / name
                if p.exists():
                    return p
            for p in d.glob("*white*.fits"):
                return p
    sciframe = os.environ.get("COMP_SCIFRAME")
    if sciframe and Path(sciframe).exists():
        return Path(sciframe).resolve()
    return None


def load_fits_image(path: Path) -> np.ndarray:
    """Load first HDU image as float array; return 2D."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"No image data in {path}")
    return np.asarray(data, dtype=np.float64).squeeze()


def run(
    main_dir: Path,
    galaxy_id: str,
    frame_id: int,
    reff: float,
    outname: str,
    initial_white_path: Path | None,
    synthetic_fits_path: Path | None,
    output_path: Path,
    circle_radius_pix: float = 18.0,
    circle_lw: float = 0.18,
    zoom_frac: float = 1.0,
    vmin_imshow: float = 2.0e-5,
    vmax_imshow: float = 0.2,
    use_tex: bool = False,
) -> None:
    main_dir = main_dir.resolve()
    white_dir = main_dir / galaxy_id / "white"
    synth_dir = white_dir / "synthetic_fits"
    coord_path = white_dir / f"white_position_{frame_id}_{outname}_reff{reff:.2f}.txt"
    match_results_path = main_dir / galaxy_id / "white" / "catalogue"
    if not (match_results_path / f"match_results_frame{frame_id}_{outname}_reff{reff:.2f}.parquet").exists():
        match_results_path = main_dir / galaxy_id / "white"
    match_parquet = match_results_path / f"match_results_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"

    fits_path = _path_env("COMP_FITS_PATH", main_dir)
    if initial_white_path is None:
        initial_white_path = find_initial_white(main_dir, galaxy_id, fits_path)
    if initial_white_path is None:
        raise FileNotFoundError(
            "Initial white-light image not found. Set COMP_SCIFRAME or pass --initial-white."
        )
    initial_white_path = Path(initial_white_path).resolve()
    if not initial_white_path.exists():
        raise FileNotFoundError(f"Initial white image not found: {initial_white_path}")

    if synthetic_fits_path is not None:
        synth_path = Path(synthetic_fits_path).resolve()
    else:
        pattern = str(synth_dir / f"*_frame{frame_id}_{outname}_reff{reff:.2f}.fits")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No synthetic FITS found: {pattern}")
        synth_path = Path(matches[0])

    if not coord_path.exists():
        raise FileNotFoundError(f"White position file not found: {coord_path}")

    # Load images (synthetic must match the coordinate frame of white_position file)
    initial = load_fits_image(initial_white_path)
    synthetic = load_fits_image(synth_path)
    ny, nx = synthetic.shape[0], synthetic.shape[1]
    # WCS for RA/Dec axes (optional)
    wcs_celestial = None
    wcs_initial = None
    try:
        with fits.open(synth_path, memmap=False) as hdul:
            _wcs = WCS(hdul[0].header)
            if _wcs.naxis >= 2:
                wcs_celestial = _wcs.celestial if _wcs.naxis > 2 else _wcs
    except Exception:
        pass
    try:
        with fits.open(initial_white_path, memmap=False) as hdul:
            _w = WCS(hdul[0].header)
            if _w.naxis >= 2:
                wcs_initial = _w.celestial if _w.naxis > 2 else _w
    except Exception:
        pass

    # Load injected positions: file from generate_white_clusters is (col, row, mag) 0-based numpy
    # (same as unravel_index: first col, second row; written as (y,x) in code = col, row)
    data = np.loadtxt(coord_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    col_0based = np.asarray(data[:, 0], dtype=np.float64)
    row_0based = np.asarray(data[:, 1], dtype=np.float64)
    # Pixel centers in imshow data coords with extent (0, nx, 0, ny): (col+0.5, row+0.5)
    x_plot = col_0based + 0.5
    y_plot = row_0based + 0.5
    # Clip to image bounds so circles are drawn only inside the displayed image
    in_bounds = (
        (col_0based >= 0) & (col_0based < nx) & (row_0based >= 0) & (row_0based < ny)
    )
    n_out = np.sum(~in_bounds)
    if n_out > 0:
        import warnings
        warnings.warn(
            f"{n_out} injected positions fall outside synthetic image shape ({ny}, {nx}); "
            "they will not be drawn. Check that coords and synthetic FITS refer to the same frame.",
            UserWarning,
            stacklevel=1,
        )

    n_injected = len(col_0based)
    # Recovered mask for panel 3: prefer full pipeline (photometry + CI) via catalogue in_catalogue;
    # fall back to white-light detection match (matched) when catalogue is missing.
    catalogue_parquet = white_dir / "catalogue" / f"catalogue_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    import pandas as pd
    if catalogue_parquet.exists():
        cat_df = pd.read_parquet(catalogue_parquet)
        if "in_catalogue" in cat_df.columns and match_parquet.exists():
            match_df = pd.read_parquet(match_parquet)
            order = match_df["cluster_id"].unique().tolist()
            cid_to_label = dict(zip(cat_df["cluster_id"], cat_df["in_catalogue"].astype(bool)))
            recovered = np.array([cid_to_label.get(cid, False) for cid in order], dtype=bool)
        else:
            recovered = np.ones(n_injected, dtype=bool)
    elif match_parquet.exists():
        df = pd.read_parquet(match_parquet)
        if "matched" in df.columns:
            recovered = np.asarray(df["matched"].astype(int).values, dtype=bool)
        else:
            recovered = np.ones(n_injected, dtype=bool)
    else:
        recovered = np.ones(n_injected, dtype=bool)
    if len(recovered) != n_injected:
        recovered = np.resize(np.asarray(recovered, dtype=bool), n_injected)

    # rc: serif, font size; use_tex=True requires system LaTeX (e.g. --usetex)
    plt.rc("font", family="serif")
    plt.rc("text", usetex=use_tex)
    plt.rcParams["font.size"] = 12

    # Single WCS from panel 1 for all three (J2000); imshow: vmin/vmax fixed, cmap Greys
    wcs_plot = wcs_initial if wcs_initial is not None else wcs_celestial
    extent_synth = (0, nx, 0, ny)
    extent_init = (0, initial.shape[1], 0, initial.shape[0])
    vmin, vmax = vmin_imshow, vmax_imshow
    cmap_imshow = "Greys"

    # FOV polygon: FK4 deg -> J2000 for black contour on all panels
    fov_ra, fov_dec = _fov_polygon_j2000()
    fov_ra_closed = np.append(fov_ra, fov_ra[0])
    fov_dec_closed = np.append(fov_dec, fov_dec[0])

    def _draw_fov(ax, use_wcs: bool):
        if use_wcs:
            ax.plot(fov_ra_closed, fov_dec_closed, "k-", lw=0.8, alpha=0.7, transform=ax.get_transform("fk5"))
        else:
            ax.plot(fov_ra_closed, fov_dec_closed, "k-", lw=1)

    def _setup_single_axis(ax, show_ylabel=True):
        """RA/Dec only: single set of axes on bottom (RA) and left (Dec); no duplicate."""
        # Fully hide the native matplotlib axis (pixel/0-1) so only WCS coords show
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # WCS: ticks and labels only on bottom (RA) and left (Dec) — none on top/right
        ax.coords[0].set_ticks_position("b")
        ax.coords[1].set_ticks_position("l")
        ax.coords[0].set_ticklabel_position("b")
        ax.coords[1].set_ticklabel_position("l")
        ax.coords[0].set_axislabel_position("b")
        ax.coords[1].set_axislabel_position("l")
        ax.coords[0].set_axislabel("Right ascension [J2000]")
        ax.coords[1].set_axislabel("Declination [J2000]" if show_ylabel else "")
        ax.coords[0].set_ticks(number=5)
        ax.coords[1].set_ticks(number=5)
        try:
            ax.coords[0].set_ticklabel(exclude_overlapping=True)
            ax.coords[1].set_ticklabel(exclude_overlapping=True)
        except Exception:
            pass
        # Ticks pointing inward (both matplotlib and WCS coordinate helpers)
        ax.tick_params(axis="both", direction="in")
        try:
            ax.coords[0].tick_params(direction="in")
            ax.coords[1].tick_params(direction="in")
        except Exception:
            pass

    # Create figure and only WCS subplots (no plt.subplots), share y (and x) axes across panels
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("white")

    if wcs_plot is not None:
        ax0 = fig.add_subplot(1, 3, 1, projection=wcs_plot)
        ax0.imshow(initial, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap_imshow)
        _setup_single_axis(ax0, show_ylabel=True)  # only left panel shows "Declination [J2000]"
        _draw_fov(ax0, use_wcs=True)
        ax1 = fig.add_subplot(1, 3, 2, projection=wcs_plot, sharex=ax0, sharey=ax0)
        ax1.imshow(synthetic, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap_imshow)
        _setup_single_axis(ax1, show_ylabel=False)
        try:
            scale_deg = float(wcs_plot.proj_plane_pixel_scales()[0].to_value("deg"))
        except Exception:
            scale_deg = 1.0 / 3600.0
        radius_deg = circle_radius_pix * scale_deg
        for i in range(n_injected):
            if not in_bounds[i]:
                continue
            xp, yp = x_plot[i], y_plot[i]
            ra, dec = wcs_plot.pixel_to_world_values(xp, yp)
            circ = Circle((ra, dec), radius_deg, fill=False, color="blue", linewidth=circle_lw, transform=ax1.get_transform("fk5"))
            ax1.add_patch(circ)
        _draw_fov(ax1, use_wcs=True)
        # Panel 3: share y (and x) with ax0
        ax2 = fig.add_subplot(1, 3, 3, projection=wcs_plot, sharex=ax0, sharey=ax0)
        ax2.imshow(synthetic, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap_imshow)
        _setup_single_axis(ax2, show_ylabel=False)
        for i in range(n_injected):
            if not in_bounds[i]:
                continue
            xp, yp = x_plot[i], y_plot[i]
            ra, dec = wcs_plot.pixel_to_world_values(xp, yp)
            if recovered[i]:
                circ = Circle((ra, dec), radius_deg, fill=False, color="lime", linewidth=circle_lw, transform=ax2.get_transform("fk5"))
                ax2.add_patch(circ)
            else:
                circ = Circle((ra, dec), radius_deg, fill=False, color="red", linewidth=circle_lw, linestyle="--", transform=ax2.get_transform("fk5"))
                ax2.add_patch(circ)
        _draw_fov(ax2, use_wcs=True)
    else:
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.imshow(initial, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap_imshow, extent=extent_init)
        ax0.set_xlim(0, initial.shape[1])
        ax0.set_ylim(0, initial.shape[0])
        ax0.set_axis_off()
        ax1 = fig.add_subplot(1, 3, 2)
        ax1.imshow(synthetic, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap_imshow, extent=extent_synth)
        ax1.set_xlim(0, nx)
        ax1.set_ylim(0, ny)
        for i in range(n_injected):
            if not in_bounds[i]:
                continue
            ax1.add_patch(Circle((x_plot[i], y_plot[i]), circle_radius_pix, fill=False, color="blue", linewidth=circle_lw))
        ax1.set_axis_off()
        ax2 = fig.add_subplot(1, 3, 3)
        ax2.imshow(synthetic, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap_imshow, extent=extent_synth)
        ax2.set_xlim(0, nx)
        ax2.set_ylim(0, ny)
        for i in range(n_injected):
            if not in_bounds[i]:
                continue
            x, y = x_plot[i], y_plot[i]
            ax2.add_patch(Circle((x, y), circle_radius_pix, fill=False, color="lime" if recovered[i] else "red", linewidth=circle_lw, linestyle="-" if recovered[i] else "--"))
        ax2.set_axis_off()

    plt.savefig(output_path, dpi=300, format=output_path.suffix.lstrip(".") or "pdf")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot white-light injection diagnostics (initial, synthetic, recovery).")
    parser.add_argument("--main-dir", type=Path, default=ROOT, help="Project root (default: repo root)")
    parser.add_argument("--galaxy", type=str, default="ngc628-c", help="Galaxy ID")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--reff", type=float, default=3.0, help="Effective radius (pc)")
    parser.add_argument("--outname", type=str, default="test", help="Output name used in pipeline")
    parser.add_argument("--initial-white", type=Path, default=None, help="Path to initial white FITS (optional)")
    parser.add_argument("--synthetic-fits", type=Path, default=None, help="Path to synthetic FITS (optional)")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: main_dir/galaxy/white/diagnostic_frame{N}_reff{R}.pdf)")
    parser.add_argument("--circle-radius", type=float, default=18.0, help="Circle radius in pixels")
    parser.add_argument("--circle-lw", type=float, default=0.18, help="Circle line width")
    parser.add_argument("--zoom-frac", type=float, default=1.0, help="Panel 3 zoom (1.0 = full image)")
    parser.add_argument("--vmin", type=float, default=2.0e-5, help="imshow vmin (default: 2e-5)")
    parser.add_argument("--vmax", type=float, default=0.2, help="imshow vmax (default: 0.2)")
    parser.add_argument("--usetex", action="store_true", help="Use LaTeX for text (requires latex in PATH)")
    args = parser.parse_args()

    main_dir = args.main_dir.resolve()
    if args.output is None:
        args.output = main_dir / args.galaxy / "white" / f"diagnostic_frame{args.frame}_reff{args.reff:.2f}.pdf"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        run(
            main_dir=main_dir,
            galaxy_id=args.galaxy,
            frame_id=args.frame,
            reff=args.reff,
            outname=args.outname,
            initial_white_path=args.initial_white,
            synthetic_fits_path=args.synthetic_fits,
            output_path=args.output,
            circle_radius_pix=args.circle_radius,
            circle_lw=args.circle_lw,
            zoom_frac=args.zoom_frac,
            vmin_imshow=args.vmin,
            vmax_imshow=args.vmax,
            use_tex=args.usetex,
        )
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
