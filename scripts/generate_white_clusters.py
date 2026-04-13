"""
Generate white-light synthetic cluster frames for the completeness pipeline.

Paths are configurable via environment variables (no hardcoded absolute paths):
  COMP_MAIN_DIR     - main run directory (default: project root)
  COMP_FITS_PATH    - LEGUS FITS images root
  COMP_PSF_PATH     - PSF files directory
  COMP_BAO_PATH     - BAOlab binary directory (contains 'bl')
  COMP_SLUG_LIB_DIR - SLUG cluster library directory
  COMP_OUTPUT_LIB_DIR - additional SLUG output library directory

CLI arguments (e.g. --directory, --fits_path) override the above.
When env and CLI are unset, paths default to the project root (directory of this file).

Placement sampling can use the white image (default), nanmean(F275W,F336W) via
--placement_mode uv_mean, or any 2D FITS matching the white frame via --placement_fits;
BAOlab still builds synthetic frames from the white --sciframe.
"""
import argparse
import datetime
import glob
import itertools
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.coordinates import Distance
from astropy.io import fits
from numpy import genfromtxt, where
from numpy.random import SeedSequence, default_rng
from scipy.ndimage import gaussian_filter

from cluster_pipeline.utils.fits_arithmetic import fits_add, fits_div

try:
    from slugpy import read_cluster
except ImportError:
    from cluster_pipeline.data.slug_reader import read_cluster  # fallback if slugpy not in env


def validate_psf_readable(psf_file: str, cam: str, filt: str) -> None:
    """Fail fast with camera/filter context if PSF file is unreadable."""
    try:
        with pyfits.open(psf_file, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError("primary HDU data is empty")
    except Exception as e:
        raise FileNotFoundError(
            f"Invalid PSF for camera={cam}, filter={filt}: {psf_file}. Reason: {e}"
        ) from e


def _as_2d_float_image(arr: Any) -> np.ndarray:
    """Squeeze FITS data to 2D float64 (placement PDF and shape checks)."""
    a = np.asarray(arr, dtype=np.float64)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {a.shape}")
    return a


def parse_exclude_regions_flat(values: list[float] | None) -> list[tuple[float, float, float]] | None:
    """
    Parse --exclude_region_param flat list into [(CX, CY, R), ...].
    CX = column (horizontal pixel index), CY = row (vertical), 0-based, same as placement (row, col).
    """
    if values is None or len(values) == 0:
        return None
    n = len(values)
    if n % 3 != 0:
        raise ValueError(
            f"--exclude_region_param must have a multiple of 3 values (CX CY R per region); got {n}"
        )
    regions: list[tuple[float, float, float]] = []
    for i in range(0, n, 3):
        cx, cy, r = float(values[i]), float(values[i + 1]), float(values[i + 2])
        if r <= 0:
            raise ValueError(
                f"--exclude_region_param: radius must be > 0, got {r} (triplet starting at index {i})"
            )
        regions.append((cx, cy, r))
    return regions


def inside_any_exclusion_region(
    row: int | float,
    col: int | float,
    exclude_regions: list[tuple[float, float, float]],
) -> bool:
    """True if (row, col) lies inside any circle (col - CX)^2 + (row - CY)^2 <= R^2."""
    fc, fr = float(col), float(row)
    for cx, cy, r in exclude_regions:
        if (fc - cx) ** 2 + (fr - cy) ** 2 <= r * r:
            return True
    return False


# -----------------------------------------------------------------------------
# Path configuration: env vars override; fallback to project root / cwd
# Set COMP_MAIN_DIR, COMP_FITS_PATH, COMP_PSF_PATH, COMP_BAO_PATH,
# COMP_SLUG_LIB_DIR, COMP_OUTPUT_LIB_DIR for custom paths.
# -----------------------------------------------------------------------------
# Repo root (scripts/generate_white_clusters.py -> parent.parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(env_key: str, default_path: Path) -> Path:
    """Resolve path from env var or default (Path)."""
    raw = os.environ.get(env_key)
    if raw:
        return Path(raw).resolve()
    return default_path.resolve() if default_path.is_absolute() else (PROJECT_ROOT / default_path).resolve()


def get_default_fits_path() -> Path:
    return _path_from_env("COMP_FITS_PATH", PROJECT_ROOT)


def get_default_psf_path() -> Path:
    return _path_from_env("COMP_PSF_PATH", PROJECT_ROOT / "PSF_all")


def get_default_bao_path() -> Path:
    return _path_from_env("COMP_BAO_PATH", PROJECT_ROOT / "baolab")


def get_default_slug_lib_dir() -> Path:
    return _path_from_env("COMP_SLUG_LIB_DIR", PROJECT_ROOT / "SLUG_library")


def get_default_output_lib_dir() -> Path:
    return _path_from_env("COMP_OUTPUT_LIB_DIR", PROJECT_ROOT / "output_lib")


##############Helper functions#############################
def phys_to_pix(args: tuple[float, float, float]) -> float:
    acpx, galdist, phys = args
    theta = np.arctan(phys / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    pix_val = theta / (acpx * u.arcsec)
    return pix_val.value


def sample_k19_radii(mass, n_draw=10):
    log_m = np.log10(mass)
    mu = 0.1415 * log_m
    sigma = 0.21

    radii_log = mu[:, None] + np.random.randn(len(mass), n_draw) * sigma
    radii_pc = 10 ** radii_log

    parent_idx = np.repeat(np.arange(len(mass)), n_draw)
    return parent_idx, radii_pc.ravel()

def load_slug_libraries_original_style(
    allfilters_cam: list[str],
    libdir: Path | str,
):
    """
    Load SLUG library using flat_in_logm_cluster_phot.fits, first file only, slugpy.read_cluster
    for L_lambda and Vega → lib_all_list, lib_all_list_veg, then concatenate with ncl_MIST=9950000.

    Returns same tuple as load_slug_libraries: (cid, actual_mass, target_mass, form_time, eval_time,
    a_v, phot_neb_ex, phot_neb_ex_veg, filter_names, filter_units). target_mass = actual_mass
    when slugpy object has no target_mass.
    """
    try:
        from slugpy import read_cluster as slugpy_read_cluster
    except ImportError as e:
        raise ImportError(
            "Original-style loading requires slugpy. Install with e.g. pip install slugpy."
        ) from e

    libdir = Path(libdir).resolve()
    lib_phot_files = glob.glob(os.path.join(libdir, "flat_in_logm_cluster_phot.fits"))
    lib_phot_files = sorted(lib_phot_files)
    if len(lib_phot_files) == 0:
        raise FileNotFoundError(
            f"No flat_in_logm_cluster_phot.fits found in {libdir}. "
            "Original-style loading expects flat_in_logm library file."
        )

    # Manually sort filter values to match the filter order of the LEGUS cluster catalogue (COL6-12)
    allfilters_cam = sorted(allfilters_cam, key=lambda x: x[-4:])

    lib_all_list = []
    lib_all_list_veg = []
    for ilib, lib in enumerate(lib_phot_files[:1]):
        libname = lib.split("_cluster_phot.fits")[0]
        print(f"Reading library clusters from file {libname}... (slugpy.read_cluster)")
        lib_read = slugpy_read_cluster(
            libname, read_filters=allfilters_cam, photsystem="L_lambda"
        )
        lib_read_vega = slugpy_read_cluster(
            libname, read_filters=allfilters_cam, photsystem="Vega"
        )
        phot_length = np.shape(lib_read.phot_neb_ex)
        print(f"library phot shape is {phot_length}")
        lib_all_list.append(lib_read)
        lib_all_list_veg.append(lib_read_vega)

    ncl_mist = 9950000
    cid = []
    actual_mass = []
    form_time = []
    eval_time = []
    a_v = []
    phot_neb_ex = []
    phot_neb_ex_veg = []
    filter_names = lib_all_list[0].filter_names
    filter_units = lib_all_list[0].filter_units

    for i, lib_all in enumerate(lib_all_list):
        cid.append(lib_all.id)
        actual_mass.append(lib_all.actual_mass)
        form_time.append(lib_all.form_time)
        eval_time.append(lib_all.time)
        a_v.append(lib_all.A_V)
        phot_neb_ex.append(lib_all.phot_neb_ex)
        phot_neb_ex_veg.append(lib_all_list_veg[i].phot_neb_ex)

    cid = np.concatenate(cid)[:ncl_mist]
    actual_mass = np.concatenate(actual_mass)[:ncl_mist]
    form_time = np.concatenate(form_time)[:ncl_mist]
    eval_time = np.concatenate(eval_time)[:ncl_mist]
    a_v = np.concatenate(a_v)[:ncl_mist]
    phot_neb_ex = np.concatenate(phot_neb_ex)[:ncl_mist, :]
    phot_neb_ex_veg = np.concatenate(phot_neb_ex_veg)[:ncl_mist, :]
    # target_mass not in original; use actual_mass so downstream shape is unchanged
    target_mass = np.array(actual_mass, dtype=float, copy=True)

    print(f"The whole library length is {np.shape(phot_neb_ex)}... \n")
    return (
        cid,
        actual_mass,
        target_mass,
        form_time,
        eval_time,
        a_v,
        phot_neb_ex,
        phot_neb_ex_veg,
        filter_names,
        filter_units,
    )


def load_slug_libraries(
    allfilters_cam: list[str],
    mrmodel: str,
    libdir: Path | str | None = None,
    output_lib_dir: Path | str | None = None,
):
    """
    Load SLUG cluster libraries and concatenate into unified arrays.

    Logic:
    - mrmodel == "flat":
        → only use flat_in_logm_cluster_phot.fits
    - otherwise (e.g. Krumholz19, Ryon17):
        → use flat_in_logm + all other *_cluster_phot.fits (excluding subsolar)

    Parameters
    ----------
    allfilters_cam : list[str]
        Filters in CAM_FILTER format, e.g. WFC3_UVIS_F275W
    mrmodel : str
        Mass-radius model name ("flat", "Krumholz19", ...)
    libdir : Path or str, optional
        Path to SLUG cluster library directory. Default from COMP_SLUG_LIB_DIR or PROJECT_ROOT/SLUG_library.
    output_lib_dir : Path or str, optional
        Path to additional output library directory. Default from COMP_OUTPUT_LIB_DIR or PROJECT_ROOT/output_lib.

    Returns
    -------
    tuple
        (cid, actual_mass, target_mass, form_time, eval_time, a_v,
         phot_neb_ex, phot_neb_ex_veg, filter_names, filter_units)
    """
    libdir = Path(libdir) if libdir is not None else get_default_slug_lib_dir()
    output_lib_dir = Path(output_lib_dir) if output_lib_dir is not None else get_default_output_lib_dir()
    libdir = libdir.resolve()
    output_lib_dir = output_lib_dir.resolve()

    # slugpy.read_cluster may omit photometry columns; use repo reader for flat concat path
    from cluster_pipeline.data.slug_reader import read_cluster as read_cluster_lib

    # --------------------------------------------------
    # 1. Decide which libraries to read
    # --------------------------------------------------
    if mrmodel == "flat":
        libs = sorted(glob.glob(str(libdir / "flat_in_logm_cluster_phot.fits")))
    else:
        libs0 = glob.glob(str(libdir / "flat_in_logm_cluster_phot.fits"))
        libs1 = glob.glob(str(output_lib_dir / "*_cluster_phot.fits"))
        libs2 = glob.glob(str(libdir / "tang*_cluster_phot.fits"))
        libsjoin = list(itertools.chain(libs0, libs1))
        libs = list(itertools.chain(libsjoin, libs2))
        libs = [lf for lf in libs if "subsolar" not in lf]
        libs = sorted(set(libs))

    if len(libs) == 0:
        raise FileNotFoundError("No SLUG library files found.")

    # --------------------------------------------------
    # 2. Sort filters to match LEGUS order
    # --------------------------------------------------
    allfilters_cam = sorted(allfilters_cam, key=lambda x: x[-4:])

    # --------------------------------------------------
    # 3. Read all libraries
    # --------------------------------------------------
    lib_all_list = []
    lib_all_list_veg = []

    for lib in libs:
        libname = lib.split("_cluster_phot.fits")[0]
        print(f"Reading library clusters from file {libname}...")

        lib_read = read_cluster_lib(
            libname,
            read_filters=allfilters_cam,
            photsystem="L_lambda",
        )
        lib_read_vega = read_cluster_lib(
            libname,
            read_filters=allfilters_cam,
            photsystem="Vega",
        )

        lib_all_list.append(lib_read)
        lib_all_list_veg.append(lib_read_vega)

    # --------------------------------------------------
    # 4. Concatenate arrays
    # --------------------------------------------------
    ncl_mist = int(1e10)  # cap total clusters to concatenate
    cid = []
    actual_mass = []
    target_mass = []
    form_time = []
    eval_time = []
    a_v = []
    phot_neb_ex = []
    phot_neb_ex_veg = []
    lib0 = lib_all_list[0]
    filter_names = getattr(lib0, "filter_names", None) or list(allfilters_cam)
    filter_units = getattr(lib0, "filter_units", None) or (["Angstrom"] * len(filter_names))

    for i, lib_all in enumerate(lib_all_list):
        cid.append(lib_all.id)
        actual_mass.append(lib_all.actual_mass)
        target_mass.append(lib_all.target_mass)
        form_time.append(lib_all.form_time)
        eval_time.append(lib_all.time)
        a_v.append(lib_all.A_V)
        phot_neb_ex.append(lib_all.phot_neb_ex)
        phot_neb_ex_veg.append(lib_all_list_veg[i].phot_neb_ex)

    cid = np.concatenate(cid)[:ncl_mist]
    actual_mass = np.concatenate(actual_mass)[:ncl_mist]
    target_mass = np.concatenate(target_mass)[:ncl_mist]
    form_time = np.concatenate(form_time)[:ncl_mist]
    eval_time = np.concatenate(eval_time)[:ncl_mist]
    a_v = np.concatenate(a_v)[:ncl_mist]
    phot_neb_ex = np.concatenate(phot_neb_ex)[:ncl_mist, :]
    phot_neb_ex_veg = np.concatenate(phot_neb_ex_veg)[:ncl_mist, :]


    # --------------------------------------------------
    # 5. Return (flat, explicit, test-friendly)
    # --------------------------------------------------
    return (
        cid,
        actual_mass,
        target_mass,
        form_time,
        eval_time,
        a_v,
        phot_neb_ex,
        phot_neb_ex_veg,
        filter_names,
        filter_units,
    )


def write_summary_log(args, total_clusters, n_train, n_val, log_path, extra=None):
    """
    Write a structured log file summarizing run parameters and key statistics.
    """
    with open(log_path, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run started: {datetime.datetime.now()}\n")
        f.write(f"Script: {os.path.basename(__file__)}\n\n")

        f.write("Arguments:\n")
        for k, v in vars(args).items():
            f.write(f"  {k:<25}: {v}\n")

        f.write("\nCluster statistics:\n")
        f.write(f"  Total clusters loaded: {total_clusters}\n")
        f.write(f"  Training clusters:     {n_train}\n")
        f.write(f"  Validation clusters:   {n_val}\n")
        f.write(f"  Validation ratio:      {n_val/(n_train or 1):.4f}\n")

        if extra:
            f.write("\nAdditional info:\n")
            for k, v in extra.items():
                f.write(f"  {k:<25}: {v}\n")

        f.write(f"\nRun completed: {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")


def safe_imarith_div_to(in_path, factor, out_path):
    fits_div(in_path, factor, out_path)


def safe_imarith_div(in_path, factor):
    fits_div(in_path, factor)


def safe_imarith_add_to(img1, img2, out_path):
    fits_add(img1, img2, out_path)


def mass_to_radius(args: tuple[Any, int | None, str]) -> np.ndarray:
    """
    Convert mass to radius using a logarithmic relationship and account for random variation.

    Args:
        args (tuple): A tuple containing:
            - libmass (float or array-like): The input mass values.
            - n_trial (int or None): The number of trials for generating random variations. If None, no variation is applied.
            - model (str): The model to use ('Krumholz19', 'Ryon17', or 'flat').

    Returns:
        numpy.ndarray: The mean radius values after applying random variation (if applicable).
    """

    libmass, n_trial, model = args
    log_libmass = np.log10(libmass)  # Compute log10(libmass)

    if model == "Krumholz19":
        # Logarithmic conversion and exponentiation
        librad = 10 ** (0.1415 * log_libmass)
        rad_lib = np.log10(librad)

        if n_trial is not None:
            # Generate random variations and calculate the mean radius
            sigma_mr = -0.2103855
            random_variations = np.random.randn(len(rad_lib), n_trial) * sigma_mr
            mean_r = rad_lib[:, None] + random_variations
            mean_r = np.mean(mean_r, axis=1)
        else:
            mean_r = rad_lib  # No random variations applied

    elif model == "Ryon17":
        # Compute librad using the OLS equation
        librad = -7.775 + 1.674 * log_libmass

        # Apply the cap: if log10(libmass) > 5.5, set librad to the capped value
        cap_value = -7.775 + 1.674 * 5.2  # Compute librad at log10(libmass) = 5.5
        librad = np.where(log_libmass > 5.2, cap_value, librad)

        mean_r = 10**librad  # Ensure consistency in return format

    elif model == "flat":
        # Return a random radius between 1 and 10 for each mass
        mean_r = np.log10(np.random.uniform(1, 10, size=len(libmass)))

    else:
        raise NotImplementedError("Model not implemented, exiting...")

    return mean_r


def clear_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")


def generate_white_light(
    scale_factors: list[float],
    f275: np.ndarray,
    f336: np.ndarray,
    f438: np.ndarray,
    f555: np.ndarray,
    f814: np.ndarray,
) -> np.ndarray:
    scale_factors = np.array(scale_factors) / scale_factors[2]
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


##############################################################
def generate_white(
    eradius: int,
    ncl: int,
    dmod: float,
    mrmodel: str,
    galaxy_fullname: str,
    validation: bool,
    overwrite: bool,
    directory: str,
    galaxy_name: str,
    outname: str,
    seed: SeedSequence,
    i_frame: int | None = None,
    i_frame_validation: int | None = None,
    framenum: int | None = None,
    make_validation_set: bool = False,
    sciframe_path: str | None = None,
    fits_path: str | None = None,
    psf_path: str | None = None,
    bao_path: str | None = None,
    slug_lib_dir: str | None = None,
    galaxy_names_list: list | None = None,
    input_coords_path: str | None = None,
    sigma_pc: float = 100.0,
    placement_mode: str = "white",
    placement_fits: str | None = None,
    exclude_regions: list[tuple[float, float, float]] | None = None,
) -> int:
    # All imports are now at the top of the file for clarity and portability
    # Set up variables as before
    directory = os.path.abspath(directory)
    main_dir = directory
    print(f"main_dir is {main_dir}")
    galaxy = galaxy_name
    galful = galaxy_fullname
    validation = validation
    overwrite = overwrite
    if validation:
        i_frame = framenum
    else:
        i_frame = i_frame if i_frame is not None else 0
    i_frame_validation = i_frame_validation if i_frame_validation is not None else 0
    framenum = framenum if framenum is not None else 0
    dmod = dmod  # distance modulus

    if fits_path is None:
        fits_path = str(get_default_fits_path())
    if psf_path is None:
        psf_path = str(get_default_psf_path())
    if bao_path is None:
        baopath = str(get_default_bao_path())
    else:
        baopath = str(bao_path)
    gal_filters = np.load(
        os.path.join(main_dir, "galaxy_filter_dict.npy"), allow_pickle=True
    ).item()
    reffs = np.array([eradius])
    pixscale_wfc3 = 0.04
    pixscale_acs = 0.05
    nums_perframe = ncl
    # Gaussian smoothing kernel for white-light footprint (physical sigma in pc)
    minsep = False

    _galaxy_names = galaxy_names_list if galaxy_names_list is not None else [galaxy_name]
    allfilters_cam = []
    for galaxy_name in _galaxy_names:
        galaxy_name = galaxy_name.split("_")[0]
        for filt, cam in zip(gal_filters[galaxy_name][0], gal_filters[galaxy_name][1]):
            filt = filt.upper()
            cam = cam.upper()
            if cam == "WFC3":
                cam = "WFC3_UVIS"
            filt_string = f"{cam}_{filt}"
            allfilters_cam.append(filt_string)
    allfilters_cam = list(set(allfilters_cam))

    # Load galaxy names and filters
    # gal_filters = np.load(
    #     os.path.join(fits_path, "galaxy_filter_dict.npy"), allow_pickle=True
    # ).item()
    filters = gal_filters[galaxy]

    # Sort filter names in desired order
    filter_names = sorted(
        filters[0]
    )  # Ensures filters are ordered, e.g., f275, f336, etc.

    # Auto-search for FITS files based on sorted filter names with assertions
    fits_files = {}
    for filter_name in filter_names:
        matches = glob.glob(f"{fits_path}/{galful}/*{filter_name}*drc.fits")
        assert len(matches) == 1, (
            f"[ERROR] Expected exactly 1 FITS file for {filter_name}, "
            f"but found {len(matches)}: {matches}"
        )
        fits_files[filter_name] = matches[0]
        print(matches[0])

    # Read FITS files dynamically based on sorted filters
    fits_data = {}
    headers = {}

    for filter_name in filter_names:
        data, hdr = fits.getdata(fits_files[filter_name], header=True)

        assert data.ndim == 2, (
            f"[ERROR] FITS data for {filter_name} is not 2D: shape={data.shape}"
        )

        fits_data[filter_name] = data
        headers[filter_name] = hdr


    # Example: Verify the order of fits_data and headers
    for f in filter_names:
        hdr = headers[f]

        assert "PHOTFLAM" in hdr, f"[ERROR] PHOTFLAM missing in header for {f}"
        assert np.isfinite(hdr["PHOTFLAM"]), f"[ERROR] PHOTFLAM not finite for {f}"
        assert hdr["PHOTFLAM"] > 0, f"[ERROR] PHOTFLAM <= 0 for {f}"

        assert hdr["NAXIS1"] > 0 and hdr["NAXIS2"] > 0, (
            f"[ERROR] Invalid image dimensions for {f}"
        )
    fname_list = [f.lower()[-5:] for f in filter_names]

    # Assert fname_list and filter_names are 1–1 aligned
    for f, fn in zip(filter_names, fname_list):
        assert fn == f

    d = Distance(distmod=dmod * u.mag)
    _ = d.to(u.cm).value  # distance_in_cm unused here; kept for future use

    _using_input_coords = input_coords_path is not None

    # --------------------------------------------------
    # Resolve data source ONCE (multiprocessing-safe)
    # (skipped entirely when using --input_coords)
    # --------------------------------------------------
    if _using_input_coords:
        pass

    elif validation:
        actual_mass_use = None
        eval_time_use   = None
        form_time_use   = None
        a_v_use         = None
        phot_neb_ex_use = None
        phot_neb_ex_veg_use = None
        mag_bao_use     = None

    elif mrmodel == "Krumholz19":
        # ---- K19: per-radius subsets ----
        if make_validation_set:
            actual_mass_use = actual_mass_val_r[eradius]
            eval_time_use   = eval_time_val_r[eradius]
            form_time_use   = form_time_val_r[eradius]
            a_v_use         = a_v_val_r[eradius]
            phot_neb_ex_use = None
            phot_neb_ex_veg_use = phot_neb_ex_veg_val_r[eradius]
            mag_bao_use     = mag_bao_val_r[eradius]
        else:
            actual_mass_use = actual_mass_main_r[eradius]
            eval_time_use   = eval_time_main_r[eradius]
            form_time_use   = form_time_main_r[eradius]
            a_v_use         = a_v_main_r[eradius]
            phot_neb_ex_use = None
            phot_neb_ex_veg_use = phot_neb_ex_veg_main_r[eradius]
            mag_bao_use     = mag_bao_main_r[eradius]

    else:
        # ---- flat / other models ----
        if make_validation_set:
            actual_mass_use = actual_mass_val
            eval_time_use   = eval_time_val
            form_time_use   = form_time_val
            a_v_use         = a_v_val
            phot_neb_ex_use = phot_neb_ex_val
            phot_neb_ex_veg_use = phot_neb_ex_veg_val
            mag_bao_use     = mag_bao_val
        else:
            actual_mass_use = actual_mass_main
            eval_time_use   = eval_time_main
            form_time_use   = form_time_main
            a_v_use         = a_v_main
            phot_neb_ex_use = phot_neb_ex_main
            phot_neb_ex_veg_use = phot_neb_ex_veg_main
            mag_bao_use     = mag_bao_main

    if _using_input_coords:
        _ic = np.loadtxt(input_coords_path)
        if _ic.ndim == 1:
            _ic = _ic.reshape(1, -1)
        assert _ic.shape[1] >= 3, (
            f"--input_coords file must have >= 3 columns (x y mag), got {_ic.shape[1]}"
        )
        print(f"[input_coords] Loaded {len(_ic)} clusters from {input_coords_path}")
        mag_bao_select = _ic[:, 2]
        ncl = len(_ic)
        if _ic.shape[1] >= 5:
            # Use mass/age from same file (1:1 with pipeline clusters)
            mass_select = np.asarray(_ic[:, 3], dtype=float)
            age_select = np.asarray(_ic[:, 4], dtype=float)
            age_select = np.maximum(age_select, 1e6)
        else:
            # No phys columns: load SLUG and cycle (not 1:1 with pipeline; prefer 5-col file)
            libdir = slug_lib_dir if slug_lib_dir else os.path.join(main_dir, "SLUG_library")
            libname = os.path.join(libdir, "flat_in_logm")
            allfilters_cam_sorted = sorted(allfilters_cam, key=lambda x: x[-4:])
            lib_slug = read_cluster(libname, read_filters=allfilters_cam_sorted, photsystem="L_lambda")
            mass_pool = np.asarray(lib_slug.actual_mass, dtype=float)
            age_pool = np.asarray(lib_slug.time, dtype=float) - np.asarray(lib_slug.form_time, dtype=float)
            age_pool = np.maximum(age_pool, 1e6)
            pool_size = len(mass_pool)
            idx = (np.arange(ncl) + i_frame * ncl) % pool_size
            mass_select = mass_pool[idx].copy()
            age_select = age_pool[idx].copy()
        av_select = np.zeros(ncl)
        mag_vega_select = np.broadcast_to(mag_bao_select.reshape(-1, 1), (ncl, 5))
        physprop_dir = os.path.join(main_dir, "physprop")
        os.makedirs(physprop_dir, exist_ok=True)
        to_save = {
            f"mass_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mass_select,
            f"age_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": age_select,
            f"av_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": av_select,
            f"mag_BAO_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mag_bao_select,
            f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mag_vega_select,
        }
        for fname, arr in to_save.items():
            savepath = os.path.join(physprop_dir, fname)
            np.save(savepath, arr)

    elif not validation:
        # --- Select corresponding properties ---
        # determine global unique offset per (eradius, frame)
        if mrmodel == "flat":
            ridx = args.eradius_list.index(eradius)
            start = (ridx * nframe + i_frame) * ncl
            end = start + ncl

            if end > len(actual_mass_use):
                raise ValueError(
                    f"Not enough clusters for eradius={eradius}: need up to {end}, have {len(actual_mass_use)}"
                )
        else:
            start = i_frame * ncl
            end   = start + ncl
        resampled_indices = np.arange(start, end)

        mass_select = actual_mass_use[resampled_indices]
        age_select  = eval_time_use[resampled_indices] - form_time_use[resampled_indices]
        av_select   = a_v_use[resampled_indices]
        mag_bao_select = mag_bao_use[resampled_indices]
        mag_vega_select = phot_neb_ex_veg_use[resampled_indices] + dmod

        assert actual_mass_use is not None
        assert mag_bao_use is not None
        if mrmodel != "Krumholz19":
            assert phot_neb_ex_use is not None

        # --- Save all arrays to .npy files ---
        physprop_dir = os.path.join(main_dir, "physprop")
        os.makedirs(physprop_dir, exist_ok=True)

        to_save = {
            f"mass_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mass_select,
            f"age_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": age_select,
            f"av_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": av_select,
            f"mag_BAO_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mag_bao_select,
            f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mag_vega_select,
        }

        for fname, arr in to_save.items():
            savepath = os.path.join(physprop_dir, fname)
            np.save(savepath, arr)

    else:
        print("Validation mode is on...")

    # Set up variables for photometry
    nums_perframe = ncl
    # Same as original_select_insert_white: maglim from actual mag_BAO range for BAOlab test frames
    _mag_bao_path = os.path.join(
        main_dir, "physprop",
        f"mag_BAO_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy",
    )
    if not validation and os.path.isfile(_mag_bao_path):
        _mag_bao = np.load(_mag_bao_path)
        maglim = [float(_mag_bao.min()), float(_mag_bao.max())]
    else:
        maglim = [-4, 4]  # fallback for validation or when physprop not yet written (e.g. input_coords)
    reff = eradius
    start_time = time.time()

    # Assigning variables to avoid confusion with the arguments
    if validation:
        galn, reff, framenum, i_frame_validation, outname = (
            galaxy_fullname,
            eradius,
            framenum,
            i_frame_validation,
            outname,
        )
    else:
        galn, reff, i_frame, outname = galaxy_fullname, eradius, i_frame, outname
    gal = galn.split("_")[0]
    gal_dir = os.path.abspath(os.path.join(main_dir, galn))
    pydir = os.path.abspath(os.path.join(gal_dir, "white"))
    path = os.path.abspath(os.path.join(pydir, "baolab"))
    rd_pattern = os.path.join(gal_dir, f"automatic_catalog*_{gal}.readme")
    matching_files = glob.glob(rd_pattern)
    if not matching_files:
        raise FileNotFoundError(f"Missing readme: pattern {rd_pattern}")
    readme_file = matching_files[0]

    with open(readme_file) as f:
        content = f.read()

    # Match aperture radius, distance modulus, and CI using regular expressions
    patterns = [
        (
            r"(?:The aperture radius used for photometry is|Photometry performed at aperture radius of)\s*(\d+(?:\.\d+)?)",
            "User-aperture radius",
        ),
        (
            r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)",
            "Galactic distance",
        ),
        (
            r"(?:This catalogue contains only )?sources with CI[ ]*>=[ ]*(\d+(?:\.\d+)?)",
            "CI value",
        ),
    ]

    for pattern, label in patterns:
        match = re.search(pattern, content)
        if match:
            if "distance" in label:
                galdist = float(match.group(2)) * 1e6
            elif "CI" in label:
                _ = float(match.group(1))  # ci
            elif "aperture" in label.lower():
                _ = float(match.group(1))  # useraperture
        else:
            raise FileNotFoundError(label + " not found in the readme.")

    # set science frame name and path
    if sciframe_path is not None:
        sciframepath = os.path.abspath(sciframe_path)
        if not os.path.exists(sciframepath):
            raise FileNotFoundError(f"Provided --sciframe not found: {sciframepath}")
        print(f"Using provided science frame: {sciframepath}")
    elif "5194" in gal:
        pattern = os.path.join(gal_dir, f"hlsp_legus_hst_*{gal}*_{filt}_*sci.fits")
        matching_files = glob.glob(pattern)
        if not matching_files:
            raise FileNotFoundError(f"No matching file for galaxy '{gal}' and filter '{filt}'")
        sciframepath = os.path.abspath(matching_files[0])
        print(f"Found matching file: {sciframepath}")
    elif "white" in galn:
        candidates = glob.glob(os.path.join(gal_dir, f"*{gal}*white*.fits"))
        if not candidates:
            candidates = glob.glob(os.path.join(gal_dir, "white_dualpop_s2n_white_remake.fits"))
        if not candidates:
            raise FileNotFoundError(
                f"No white-light science frame found in {gal_dir}. "
                "Use --sciframe to specify the path explicitly."
            )
        sciframepath = os.path.abspath(candidates[0])
        print(f"Found matching file: {sciframepath}")
    else:
        raise FileNotFoundError("Cannot determine science frame. Use --sciframe to specify.")

    hd = pyfits.getheader(sciframepath)

    fname = filt  # set filter name
    # PSF files use psf_*_<camera>_<filter>.fits with lowercase camera stem + filter (same as inject_clusters_to_5filters.py)
    cam_lower = (str(cam).lower() if cam else "").split("_")[0]
    filt_lower = str(filt).lower()
    psfname = glob.glob(os.path.join(psf_path, f"psf_*_{cam_lower}_{filt_lower}.fits"))
    if not psfname:
        psfname = glob.glob(os.path.join(psf_path, f"psf_*_{cam}_{filt}.fits"))
    if psfname:
        print(f"Found PSF file: {psfname}")
    elif "white" in galn or sciframe_path is not None:
        raise FileNotFoundError(
            f"No PSF found matching psf_*_{cam_lower}_{filt_lower}.fits (or psf_*_{cam}_{filt}.fits) in {psf_path} for white injection."
        )
    else:
        raise FileNotFoundError(
            f"No PSF found matching psf_*_{cam_lower}_{filt_lower}.fits in {psf_path}"
        )
    psffile = psfname[0]
    validate_psf_readable(psffile, str(cam), str(filt))
    psffilepath = psffile  # this path contains all PSF files (wfc3 and acs)

    # set bao_fhwm for photometry
    print("reff is", reff)
    print("galdist is", galdist)
    print("pixscale_wfc3 is", pixscale_wfc3)
    pixscale_arcsec = pixscale_wfc3 if ("wfc3" in str(cam).lower()) else pixscale_acs
    bao_fhwm = (reff / galdist) * (180.0 / np.pi) * (3600.0 / pixscale_arcsec) / 1.13

    print(bao_fhwm)

    # ---- --- -- - BAOlab zeropoint value: - -- --- ----
    baozpoint = 1e3
    #     baozpoint = 1 ##### CHANGE OF ZP for white light!

    # ---- --- -- - scientific frame dimensions: - -- --- ----
    hd = pyfits.getheader(sciframepath)
    xaxis = hd["NAXIS1"]
    yaxis = hd["NAXIS2"]

    # ---- --- -- - finding the zeropoint and exptime - -- --- ----
    zpfile = os.path.join(gal_dir, f"header_info_{gal}.txt")
    filters, instrument, zpoint, exptime_col = np.loadtxt(
        zpfile, unpack=True, skiprows=0, usecols=(0, 1, 2, 3), dtype="str"
    )
    match = where(np.char.lower(filters.astype(str)) == str(fname).lower())
    # Set zp to be 1 for easy computation
    zp = 1
    # White frames generated via averaging may not preserve EXPTIME in header.
    # Fall back to per-filter exposure listed in header_info_<gal>.txt.
    try:
        expt = hd["EXPTIME"]
    except KeyError:
        expt = float(exptime_col[match][0])
        print(f"EXPTIME not found in science header; using header_info value: {expt}")
    print("zp: ")
    print(zp)
    print("expt: ")
    print(expt)
    # --------------------------
    # STEP 1: SOURCE CREATION
    # --------------------------

    # Clear or create the baolab directory
    clear_directory(path)

    # Clear or create the synthetic_frames directory
    clear_directory(os.path.join(pydir, "synthetic_frames"))

    # Clear or create the synthetic_fits directory
    clear_directory(os.path.join(pydir, "synthetic_fits"))

    #########################################################################################
    ######### I  GENERATE 2 TEST SOURCES FOR A VISUAL CHECK OF THE SIMULATED SOURCES ########
    #########################################################################################
    # path = pydir + "/baolab"
    # os.chdir(path)
    # factor_zp=10**(-0.4*zp)
    factor_def = baozpoint

    # name a unique .bl file to avoid multi-core overstepping.
    if validation:
        mk_cmppsf_bl = f"mk_cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.bl"
        cmppsf_r0 = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}_test{reffs[0]:.2f}pc.fits"
        cmppsf_r1 = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}_test{reffs[-1]:.2f}pc.fits"
        cmppsf_r = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.fits"

        f = open(os.path.join(path, mk_cmppsf_bl), "w")
        f.write("# I GENERATE THE COMPOSITE PSFs \n\n")
        f.write(
            "mkcmppsf "
            + cmppsf_r
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(bao_fhwm)
            + " MKCMPPSF.FWHMOBJY="
            + str(bao_fhwm)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r0
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(bao_fhwm)
            + " MKCMPPSF.FWHMOBJY="
            + str(bao_fhwm)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r1
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(bao_fhwm)
            + " MKCMPPSF.FWHMOBJY="
            + str(bao_fhwm)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write("quit \n")
        f.close()

        testcoord1 = (
            f"testcoord1_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}"
            + ".txt"
        )
        g1 = open(os.path.join(path, testcoord1), "w")
        g1.write("50 50 " + str(maglim[0]))
        g1.close()

        testcoord2 = (
            f"testcoord2_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}"
            + ".txt"
        )
        g2 = open(os.path.join(path, testcoord2), "w")
        g2.write("50 50 " + str(maglim[-1]))
        g2.close()

        testimg1 = (
            "test_source_mag_"
            + f"{maglim[0]:.1f}{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}"
            + f"_test{reffs[0]:.2f}.fits"
        )
        testimg2 = (
            "test_source_mag_"
            + f"{maglim[1]:.1f}{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}"
            + f"_test{reffs[-1]:.2f}.fits"
        )

        mk_test_bl = f"mk_test_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.bl"
        f = open(os.path.join(path, mk_test_bl), "w")
        f.write(
            "# I GENERATE THE TEST SOURCE IN ORDER TO GET THE REAL ZEROPOINT I NEED \n\n"
        )
        f.write(
            "mksynth "
            + testcoord1
            + " "
            + testimg1
            + " MKSYNTH.REZX=100 MKSYNTH.REZY=100 MKSYNTH.RANDOM=NO MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT="
            + str(baozpoint)
            + " MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE="
            + cmppsf_r0
            + " \n \n"
        )
        f.write(
            "mksynth "
            + testcoord2
            + " "
            + testimg2
            + " MKSYNTH.REZX=100 MKSYNTH.REZY=100 MKSYNTH.RANDOM=NO MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT="
            + str(baozpoint)
            + " MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE="
            + cmppsf_r1
            + " \n \n"
        )
        f.write("quit \n")
        f.close()

        print("Running baolab to create the test frames...")
        cmppsf_log = f"bao_reff{reff:.2f}_cmppsf_frame{i_frame}_{outname}_vframe{i_frame_validation}.txt"
        test_log = f"bao_reff{reff:.2f}_test_frame{i_frame}_{outname}_vframe{i_frame_validation}.txt"
        # Run mk_cmppsf_bl
        with (
            open(os.path.join(path, mk_cmppsf_bl)) as script,
            open(os.path.join(path, cmppsf_log), "w") as log,
        ):
            subprocess.run(
                [os.path.join(baopath, "bl")], stdin=script, stdout=log, cwd=path, check=True
            )

        # Run mk_test_bl
        with (
            open(os.path.join(path, mk_test_bl)) as script,
            open(os.path.join(path, test_log), "w") as log,
        ):
            subprocess.run(
                [os.path.join(baopath, "bl")], stdin=script, stdout=log, cwd=path, check=True
            )

        # iraf.imarith(operand1 = testimg1, op = '/', operand2 = factor_def, result = testimg1)
        # iraf.imarith(operand1 = testimg2, op = '/', operand2 = factor_def, result = testimg2)
        safe_imarith_div(os.path.join(path, testimg1), factor_def)
        safe_imarith_div(os.path.join(path, testimg2), factor_def)

        if os.path.exists(testimg1):
            print(f"Generated image {testimg1}.")
        if os.path.exists(testimg2):
            print(f"Generated image {testimg2}.")
    else:
        mk_cmppsf_bl = (
            f"mk_cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}.bl"
        )
        cmppsf_r0 = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}_test{reffs[0]:.2f}pc.fits"
        cmppsf_r1 = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}_test{reffs[-1]:.2f}pc.fits"
        cmppsf_r = (
            f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}.fits"
        )

        f = open(os.path.join(path, mk_cmppsf_bl), "w")
        f.write("# I GENERATE THE COMPOSITE PSFs \n\n")
        f.write(
            "mkcmppsf "
            + cmppsf_r
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(bao_fhwm)
            + " MKCMPPSF.FWHMOBJY="
            + str(bao_fhwm)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r0
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(bao_fhwm)
            + " MKCMPPSF.FWHMOBJY="
            + str(bao_fhwm)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r1
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(bao_fhwm)
            + " MKCMPPSF.FWHMOBJY="
            + str(bao_fhwm)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write("quit \n")
        f.close()

        testcoord1 = (
            f"testcoord1_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}"
            + ".txt"
        )
        g1 = open(os.path.join(path, testcoord1), "w")
        g1.write("50 50 " + str(maglim[0]))
        g1.close()

        testcoord2 = (
            f"testcoord2_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}"
            + ".txt"
        )
        g2 = open(os.path.join(path, testcoord2), "w")
        g2.write("50 50 " + str(maglim[-1]))
        g2.close()

        testimg1 = (
            "test_source_mag_"
            + f"{maglim[0]:.1f}{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}"
            + f"_test{reffs[0]:.2f}.fits"
        )
        testimg2 = (
            "test_source_mag_"
            + f"{maglim[1]:.1f}{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}"
            + f"_test{reffs[-1]:.2f}.fits"
        )

        mk_test_bl = (
            f"mk_test_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}.bl"
        )
        f = open(os.path.join(path, mk_test_bl), "w")
        f.write(
            "# I GENERATE THE TEST SOURCE IN ORDER TO GET THE REAL ZEROPOINT I NEED \n\n"
        )
        f.write(
            "mksynth "
            + testcoord1
            + " "
            + testimg1
            + " MKSYNTH.REZX=100 MKSYNTH.REZY=100 MKSYNTH.RANDOM=NO MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT="
            + str(baozpoint)
            + " MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE="
            + cmppsf_r0
            + " \n \n"
        )
        f.write(
            "mksynth "
            + testcoord2
            + " "
            + testimg2
            + " MKSYNTH.REZX=100 MKSYNTH.REZY=100 MKSYNTH.RANDOM=NO MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT="
            + str(baozpoint)
            + " MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE="
            + cmppsf_r1
            + " \n \n"
        )
        f.write("quit \n")
        f.close()

        # Run mk_cmppsf_bl
        print("Running baolab to create the test frames...")

        cmppsf_log = f"bao_reff{reff:.2f}_cmppsf_frame{i_frame}_{outname}.txt"
        test_log = f"bao_reff{reff:.2f}_test_frame{i_frame}_{outname}.txt"

        # Run mk_cmppsf_bl
        with (
            open(os.path.join(path, mk_cmppsf_bl)) as script,
            open(os.path.join(path, cmppsf_log), "w") as log,
        ):
            subprocess.run(
                [os.path.join(baopath, "bl")], stdin=script, stdout=log, cwd=path, check=True
            )

        # Run mk_test_bl
        with (
            open(os.path.join(path, mk_test_bl)) as script,
            open(os.path.join(path, test_log), "w") as log,
        ):
            subprocess.run(
                [os.path.join(baopath, "bl")], stdin=script, stdout=log, cwd=path, check=True
            )

        testimg1_path = os.path.join(path, testimg1)
        testimg2_path = os.path.join(path, testimg2)

        safe_imarith_div(testimg1_path, factor_def)
        safe_imarith_div(testimg2_path, factor_def)

        # File existence check also needs full paths
        if os.path.exists(testimg1_path):
            print(f"Generated image {testimg1_path}.")
        if os.path.exists(testimg2_path):
            print(f"Generated image {testimg2_path}.")

    #########################################################################################
    ######### I  GENERATE THE COMPOSITE THE SYNTHETIC SOURCES FOR COMPLETENESS TEST #########
    #########################################################################################
    print(
        "I am writing the baolab file to generate the frames with synthetic sources..."
    )
    # Load white-light FITS (BAOlab mksynth input); placement PDF may use another map below.
    hdul = fits.open(sciframepath)
    image_data = hdul[0].data
    white_image_2d = _as_2d_float_image(image_data)

    if placement_fits:
        pmap_path = os.path.abspath(placement_fits)
        if not os.path.isfile(pmap_path):
            raise FileNotFoundError(f"--placement_fits not found: {pmap_path}")
        with fits.open(pmap_path) as pm_hdu:
            placement_image = _as_2d_float_image(pm_hdu[0].data)
        if placement_image.shape != white_image_2d.shape:
            raise ValueError(
                f"placement_fits shape {placement_image.shape} != white science frame "
                f"{white_image_2d.shape} ({sciframepath})"
            )
        print(f"[placement] Using --placement_fits for PDF: {pmap_path}")
    elif placement_mode == "uv_mean":
        f275_keys = [fn for fn in filter_names if fn.lower().startswith("f275")]
        f336_keys = [fn for fn in filter_names if fn.lower().startswith("f336")]
        if not f275_keys or not f336_keys:
            raise ValueError(
                "placement_mode=uv_mean requires F275W and F336W in galaxy_filter_dict / "
                f"filter_names; got {filter_names!r}"
            )
        k275, k336 = f275_keys[0], f336_keys[0]
        placement_image = np.nanmean(
            np.stack([fits_data[k275], fits_data[k336]], axis=0), axis=0
        ).astype(np.float64, copy=False)
        if placement_image.shape != white_image_2d.shape:
            raise ValueError(
                f"UV mean shape {placement_image.shape} != white science frame "
                f"{white_image_2d.shape}"
            )
        print(f"[placement] Using nanmean({k275}, {k336}) for PDF (sigma_pc={sigma_pc} pc)")
    elif placement_mode == "white":
        placement_image = white_image_2d
    else:
        raise ValueError(f"Unknown placement_mode {placement_mode!r}; use white or uv_mean")

    image_data_flat = placement_image.flatten()

    # convert physical size to pixel size

    theta = np.arctan(sigma_pc * u.pc / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    if cam == "wfc3" or "wfc3" in cam.lower():
        pix_scale = pixscale_wfc3
    elif cam == "acs" or "acs" in cam.lower():
        pix_scale = pixscale_acs
    elif "white" in galn or sciframe_path is not None:
        pix_scale = pixscale_wfc3
    else:
        raise TypeError(f"Unknown camera type '{cam}', exiting..")

    # convert to pixel scale and gaussian-convolve with sigma = pixel scale
    pix_cl = (sigma_pc / (2 * np.pi * galdist) * 3600) / pix_scale
    # convert pc scale to pixel-scale for gaussian_filter function
    filtered = gaussian_filter(placement_image, sigma=pix_cl)
    filtered_flat = filtered.flatten()

    # Get indices where values are greater than 0
    positive_indices_image = np.where(image_data_flat > 0)
    positive_indices_filtered = np.where(filtered_flat > 0)

    # Create boolean arrays with the same shape as the original arrays
    image_data_positive = np.zeros_like(image_data_flat, dtype=bool)
    filtered_positive = np.zeros_like(filtered_flat, dtype=bool)

    # Mark the positive indices as True
    image_data_positive[positive_indices_image] = True
    filtered_positive[positive_indices_filtered] = True

    # Element-wise logical AND operation (same as original_select_insert_white.py)
    p_id = np.logical_and(image_data_positive, filtered_positive)
    positive_indices_result = where(p_id)[0]
    image_shape = placement_image.shape

    # PDF for cluster placement: proportional to (convolved) light profile on allowed pixels
    placement_weights = np.asarray(filtered_flat[positive_indices_result], dtype=np.float64)
    placement_weights = np.maximum(placement_weights, 0.0)
    if placement_weights.sum() <= 0:
        raise ValueError("No positive flux in convolved image on allowed mask; cannot define placement PDF.")
    placement_weights /= placement_weights.sum()

    # Minimum separation: 3 times the effective radius in pixels (same as original_select_insert_white.py)
    theta = np.arctan(reff * 3 * u.pc / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    pix_cl = theta / (pix_scale * u.arcsec)
    min_separation = 3 * pix_cl.value

    # Record the start time
    start_time = time.time()

    # Initialize the list to store selected coordinates
    if _using_input_coords:
        selected_coordinates = [(int(row[0]), int(row[1])) for row in _ic]
        print(f"[input_coords] Using {len(selected_coordinates)} positions from file")
    else:
        selected_coordinates = []
        rng = default_rng(seed)  # seed is a SeedSequence object passed from main()
        if exclude_regions:
            print(
                f"[exclude_region_param] {len(exclude_regions)} circular exclusion(s) "
                "(CX=column, CY=row, 0-based pixels)"
            )
        max_attempts = (
            max(10_000_000, nums_perframe * 1_000_000) if exclude_regions else None
        )
        attempts = 0

        while len(selected_coordinates) < nums_perframe:
            if max_attempts is not None:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f"Could not sample {nums_perframe} injection positions after {max_attempts} tries; "
                        "exclusion circles may cover the placement mask — relax regions or mask."
                    )
            # Sample position from PDF proportional to (sigma_pc-convolved) light profile
            random_coord = rng.choice(positive_indices_result, p=placement_weights)

            # Convert the random coordinate to (row, col)
            row, col = np.unravel_index(random_coord, image_shape)

            if exclude_regions is not None and inside_any_exclusion_region(
                row, col, exclude_regions
            ):
                continue

            if (row, col) in selected_coordinates:
                continue

            # Flag to check if the coordinate meets the minimum separation requirement
            if minsep:
                meets_requirement = True

                if len(selected_coordinates) > 0:
                    # Check the distance to the nearest neighbor for each selected coordinate
                    row_sel = np.array([coord[0] for coord in selected_coordinates])
                    col_sel = np.array([coord[1] for coord in selected_coordinates])
                    distance = np.sqrt((row - row_sel) ** 2 + (col - col_sel) ** 2)
                    if np.any(distance < min_separation):
                        meets_requirement = False

                if not meets_requirement:
                    continue  # Restart the loop for a new random coordinate
                else:
                    selected_coordinates.append((row, col))
            else:
                selected_coordinates.append((row, col))

    # --- 1) Build filenames ---
    if not validation:
        mk_frames_bl = (
            f"mk_frames_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}.bl"
        )
        namecoord = (
            f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{i_frame}_{outname}.txt"
        )
        namesource = f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{i_frame}_{outname}_temp.fits"
    else:
        mk_frames_bl = f"mk_frames_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}_validation.bl"
        namecoord = f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation.txt"
        namesource = f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_temp.fits"

    # --- 2) Non-validation mode: write filter mags and white files ---
    if not validation:
        bao_mag_select = mag_bao_select

        # 2a) Filter-specific magnitude files (skip when using input_coords)
        if not _using_input_coords:
            for ifn, fname in enumerate(filter_names):
                fn = os.path.join(
                    pydir, f"{fname}_position_{i_frame}_{outname}_reff{reff:.2f}.txt"
                )
                if not os.path.exists(fn):
                    with open(fn, "w") as fh:
                        for (y, x), mag in zip(
                            selected_coordinates, mag_vega_select[:, ifn]
                        ):
                            fh.write(f"{y} {x} {mag}\n")
                else:
                    print(f"Filter mag file exists: {fn}")

        # 2b) White + coord file
        white_fn = os.path.join(
            pydir, f"white_position_{i_frame}_{outname}_reff{reff:.2f}.txt"
        )
        if os.path.exists(white_fn) and not _using_input_coords:
            print(f"White file exists; copying to {namecoord}")
            shutil.copy(white_fn, os.path.join(path, namecoord))
        else:
            print("Creating white + coord files.")
            with (
                open(white_fn, "w") as wf,
                open(os.path.join(path, namecoord), "w") as nc,
            ):
                for (y, x), mag in zip(selected_coordinates, bao_mag_select):
                    wf.write(f"{y} {x} {mag}\n")
                    nc.write(f"{y} {x} {mag}\n")

    # --- 3) Validation mode: reload mags and write validation files ---
    else:
        # 3a) Load BAO mags from existing white file
        white_orig = os.path.join(
            pydir, f"white_position_{i_frame}_{outname}_reff{reff:.2f}.txt"
        )  # TESTING
        print(f"TEMPORARY WHITE FILE: {white_orig}")
        if not os.path.exists(white_orig):
            raise FileNotFoundError(f"Missing validation white file: {white_orig}")

        bao_mag_select = np.loadtxt(white_orig)[:, -1]
        mag_vega_select = np.zeros((len(bao_mag_select), len(filter_names)))

        # 3b) Rebuild mag_vega_select from filter files
        for ifn, fname in enumerate(filter_names):
            fn = os.path.join(
                pydir, f"{fname}_position_{i_frame}_{outname}_reff{reff:.2f}.txt"
            )
            mag_vega_select[:, ifn] = np.loadtxt(fn)[:, -1]

        # 3c) Write validation filter files
        for ifn, fname in enumerate(filter_names):
            fn_val = os.path.join(
                pydir,
                f"{fname}_position_{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
            )
            if not os.path.exists(fn_val):
                with open(fn_val, "w") as fh:
                    for (y, x), mag in zip(
                        selected_coordinates, mag_vega_select[:, ifn]
                    ):
                        fh.write(f"{y} {x} {mag}\n")
            else:
                print(f"Validation filter file exists: {fn_val}")

        # 3d) Validation white + coord file
        white_val = os.path.join(
            pydir,
            f"white_position_{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
        )
        if os.path.exists(white_val):
            print(f"Validation white exists; copying to {namecoord}")
            shutil.copy(white_val, os.path.join(path, namecoord))
        else:
            print(f"Creating validation white + coord files: {white_val}")

            with (
                open(white_val, "w") as wf,
                open(os.path.join(path, namecoord), "w") as nc,
            ):
                for (y, x), mag in zip(selected_coordinates, bao_mag_select):
                    wf.write(f"{y} {x} {mag}\n")
                    nc.write(f"{y} {x} {mag}\n")

    #########################################################################################
    ######### I  GENERATE THE COMPOSITE THE SYNTHETIC SOURCES FOR COMPLETENESS TEST #########
    #########################################################################################

    f = open(os.path.join(path, mk_frames_bl), "w")
    f.write(
        "# I GENERATE THE TEST SOURCE IN ORDER TO GET THE REAL ZEROPOINT I NEED \n \n"
    )
    f.write(
        "mksynth "
        + namecoord
        + " "
        + namesource
        + " MKSYNTH.REZX="
        + str(xaxis)
        + " MKSYNTH.REZY="
        + str(yaxis)
        + " MKSYNTH.RANDOM=NO MKSYNTH.NSTARS=100 MKSYNTH.MAGFAINT=0. MKSYNTH.MAGBRIGHT=0. MKSYNTH.MAGSTEP=0 MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT="
        + str(baozpoint)
        + " MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE="
        + cmppsf_r
        + " \n \n"
    )
    f.write("quit \n")
    f.close()
    print("Name source is...", namesource)
    print("Name coord is...", namecoord)
    print("Making synthetic sources on", namesource)

    # Define paths
    baolab_dir = os.path.join(
        pydir, "baolab"
    )  # Replace with your actual BAOlab working directory if different
    mk_frames_bl_path = os.path.join(baolab_dir, mk_frames_bl)
    bao_log_path = os.path.join(baolab_dir, f"bao_{i_frame}_{outname}.txt")

    # Run BAOlab with subprocess in specified directory
    with open(mk_frames_bl_path) as script, open(bao_log_path, "w") as log:
        subprocess.run(
            [os.path.join(baopath, "bl")], stdin=script, stdout=log, cwd=baolab_dir, check=False
        )

    # Safely copy all *pc_sources_*.txt to synthetic_fits
    src_pattern = os.path.join(baolab_dir, namecoord)
    dest_dir = os.path.join(pydir, "synthetic_fits")

    for src_file in glob.glob(src_pattern):
        dest_path = os.path.join(dest_dir, os.path.basename(src_file))
        shutil.copy(src_file, dest_path)
    print("Copied synthetic clusters on fits image to fits directory.")

    ##########################################################################################
    ########### I  DIVIDE THE FRAMES FOR THE CORRECT FACTOR AND ADD THE BACKGROUND ###########
    ##########################################################################################
    print("")
    print("I am adding the newly generated synthetic frames to the background...")
    print(
        "The resulting frames of this operation will be the ones to use for completeness test!"
    )
    print("But this operation can take several minutes! BE PATIENT (again!!!)...")

    # 1. Construct list filename
    if not validation:
        listfile = os.path.join(
            path,
            f"list_temp_{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}.txt",
        )
    else:
        listfile = os.path.join(
            path,
            f"list_temp_{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
        )

    # 2. Pipe `namesource` (e.g., "*.fits") to list file
    with open(listfile, "w") as f:
        subprocess.run(f"ls {namesource}", cwd=path, shell=True, stdout=f, check=True)

    # 3. Load FITS list
    listimg = genfromtxt(listfile, dtype="str")

    if isinstance(listimg, str) or listimg.ndim == 0:
        print(f"Only one FITS file found: {listimg}")
        listimg = [str(listimg)]  # convert to list of strings
    elif listimg.ndim == 1:
        print(f"{len(listimg)} FITS files found.")
        listimg = list(map(str, listimg))  # ensure all are strings
    else:
        raise ValueError("Unexpected listimg format.")
    # 4. Loop over all images
    for i in range(len(listimg)):
        print(f"image {i + 1} out of {len(listimg)}")

        if not validation:
            name_final = os.path.join(
                path, f"{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}.fits"
            )
            temp2 = os.path.join(
                path,
                f"{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}_temp2.fits",
            )
        else:
            name_final = os.path.join(
                path,
                f"{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.fits",
            )
            temp2 = os.path.join(
                path,
                f"{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}_temp2.fits",
            )

        print("_image string from index 0 to -10 is", listimg[i])
        print("final name is", name_final)
        # Ensure input path is full
        print(f"Image path :{path} \n")
        print(f"Image list is :{listimg[i]} \n")
        print(f"Iteration {i}------- \n")
        lism = listimg[i]
        operand1_path = os.path.join(path, lism)

        # Clean up old temp2 if it exists
        if os.path.exists(temp2):
            os.remove(temp2)

        safe_imarith_div_to(operand1_path, factor_def, temp2)
        safe_imarith_add_to(temp2, sciframepath, name_final)

        try:
            shutil.copy(name_final, os.path.join(pydir, "synthetic_fits"))
        except OSError as e:
            raise FileNotFoundError(
                f"{name_final} is not found or failed to copy to synthetic_fits"
            ) from e

    ######################################################################################################
    ################ MAKE SOME ORDER BEFORE STARTING WITH EXTRACTION AND PHOTOMETRY !!!!! ################
    ######################################################################################################

    # move synthetic frames
    # os.chdir(pydir+'/baolab/')
    # path = pydir+'/synthetic_fits/'
    # os.chdir(path)
    # os.chdir(pydir)

    print("\n Source creation is completed! \n")

    # End time
    end_time = time.time()

    # Compute elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser(description="Process galaxies in batches.")
    parser.add_argument(
        "--gal_name", type=str, default=None, help="name of the galaxy (str)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Main directory for the run (default: COMP_MAIN_DIR env or current working directory)",
    )
    parser.add_argument(
        "--ncl", type=int, default=500, help="number of star clusters (int)"
    )
    parser.add_argument(
        "--dmod",
        type=float,
        default=29.98,
        help="distance modulus of target galaxy (float)",
    )
    parser.add_argument(
        "--mrmodel",
        type=str,
        default="flat",
        help="Mass-radius relation to compute effective radii (str)",
    )
    parser.add_argument(
        "--eradius_list",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="List of eradius values (list of int)",
    )
    parser.add_argument(
        "--nframe", type=int, default=None, help="Number of frames (int, optional)"
    )
    parser.add_argument(
        "--nframe_validation",
        type=int,
        default=20,
        help="Number of validation set frames (int, optional)",
    )
    parser.add_argument(
        "--framenum",
        type=int,
        default=0,
        help="Frame number to perform validation on (int, optional)",
    )
    parser.add_argument(
        "--galaxy_fullname",
        type=str,
        default=None,
        help="full name/path key of the galaxy directory; defaults to --gal_name when omitted",
    )
    parser.add_argument(
        "--outname", type=str, default=None, help="output name (str, optional)"
    )
    parser.add_argument(
        "--nframe_validation_start",
        type=int,
        default=0,
        help="Starting frame number for validation (int)",
    )
    parser.add_argument(
        "--nframe_validation_end",
        type=int,
        default=100,
        help="Ending frame number for validation (int)",
    )
    parser.add_argument(
        "--sciframe",
        type=str,
        default=None,
        help="Path to the white-light science frame FITS file (e.g. ngc628-c_white.fits)",
    )
    parser.add_argument(
        "--fits_path",
        type=str,
        default=None,
        help="Root path for LEGUS FITS images (default: same as --directory, or COMP_FITS_PATH env, or project root)",
    )
    parser.add_argument(
        "--psf_path",
        type=str,
        default=None,
        help="Path to PSF files directory (default: COMP_PSF_PATH env or project_root/PSF_all)",
    )
    parser.add_argument(
        "--bao_path",
        type=str,
        default=None,
        help="Path to BAOlab binary directory containing 'bl' (default: COMP_BAO_PATH env or project_root/baolab)",
    )
    parser.add_argument(
        "--slug_lib_dir",
        type=str,
        default=None,
        help="Path to SLUG cluster library directory (default: COMP_SLUG_LIB_DIR env or project_root/SLUG_library)",
    )
    parser.add_argument(
        "--input_coords",
        type=str,
        default=None,
        help="Path to a 3-column (x y mag) coordinate file. "
             "When provided, clusters are injected at these exact positions "
             "with these magnitudes, bypassing SLUG library sampling.",
    )
    parser.add_argument(
        "--sigma_pc",
        type=float,
        default=100.0,
        help="Gaussian smoothing kernel for placement footprint in pc (default 100); "
        "applies to the map chosen by --placement_mode / --placement_fits.",
    )
    parser.add_argument(
        "--placement_mode",
        type=str,
        default="white",
        choices=["white", "uv_mean"],
        help="Map for cluster placement PDF: white (same as --sciframe) or uv_mean "
        "(nanmean of F275W and F336W drizzled images under --fits_path). Ignored if "
        "--placement_fits is set.",
    )
    parser.add_argument(
        "--placement_fits",
        type=str,
        default=None,
        help="Optional FITS path for placement PDF only; pixel shape must match --sciframe. "
        "BAOlab still injects on the white science frame. Overrides --placement_mode.",
    )
    parser.add_argument(
        "--exclude_region_param",
        nargs="+",
        type=float,
        default=None,
        help="Circular exclusion zones for injection sampling only (pixels): "
        "CX1 CY1 R1 CX2 CY2 R2 ... with CX=column, CY=row (0-based). Reject and resample if inside any circle.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Enable overwriting existing files",
    )
    parser.add_argument(
        "--validation",
        default=False,
        action="store_true",
        help="Run in validation mode",
    )
    parser.add_argument(
        "--make_validation_set",
        default=False,
        action="store_true",
        help="Make validation set for later use.",
    )
    parser.add_argument(
        "--auto_check_missing", action="store_true", help="Auto check missing files"
    )
    args = parser.parse_args()

    try:
        exclude_regions_parsed = parse_exclude_regions_flat(args.exclude_region_param)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # Default --directory from env or project root (no hardcoded paths)
    if args.directory is None:
        args.directory = os.environ.get("COMP_MAIN_DIR", str(PROJECT_ROOT))
    args.directory = os.path.abspath(args.directory)

    eradius_list = args.eradius_list
    ncl = args.ncl
    dmod = args.dmod
    mrmodel = args.mrmodel
    directory = args.directory
    galaxy_fullname = args.galaxy_fullname or args.gal_name
    galaxy_name = args.gal_name
    outname = args.outname
    validation = args.validation
    overwrite = args.overwrite
    nframe = args.nframe
    nframe_validation_start = args.nframe_validation_start
    nframe_validation_end = args.nframe_validation_end
    auto_check_missing = args.auto_check_missing
    make_validation_set = args.make_validation_set
    #########READ IN SLUG LIBRARY#######################
    fits_path = args.fits_path or args.directory
    galaxy_names = [args.gal_name]
    galaxies = np.load(os.path.join(args.directory, "galaxy_names.npy"))
    gal_filters = np.load(
        os.path.join(args.directory, "galaxy_filter_dict.npy"), allow_pickle=True
    ).item()
    allfilters_cam = []

    for galaxy_name in galaxy_names:
        galaxy_name = galaxy_name.split("_")[0]
        for filt, cam in zip(gal_filters[galaxy_name][0], gal_filters[galaxy_name][1]):
            filt = filt.upper()
            cam = cam.upper()
            if cam == "WFC3":
                cam = "WFC3_UVIS"
            filt_string = f"{cam}_{filt}"
            allfilters_cam.append(filt_string)
    allfilters_cam = list(set(allfilters_cam))
    print(allfilters_cam)
    # --------------------------------------------------
    # Load SLUG libraries (centralised, model-aware)
    # --------------------------------------------------
    if args.input_coords is not None:
        print("[input_coords] Skipping SLUG library loading — using user-supplied coordinates.")
        train_blocks = args.nframe if args.nframe is not None else 1
        val_blocks = 0

    elif not args.validation:

        (
            cid,
            actual_mass,
            target_mass,
            form_time,
            eval_time,
            a_v,
            phot_neb_ex,
            phot_neb_ex_veg,
            filter_names,
            filter_units,
        ) = load_slug_libraries(
            allfilters_cam=allfilters_cam,
            mrmodel=mrmodel,
            libdir=args.slug_lib_dir,
        )
        print(f"[LIB] Loaded {len(actual_mass)} clusters total (slugpy.read_cluster).")

        # ------------------------------------------------------------------
        # Apply brightness cutoff BEFORE splitting into subsets
        #    Remove clusters with any filter magnitude < 14.2 mag to avoid
        #   unphysically bright clusters and speed up baolab processing
        # ------------------------------------------------------------------
        M_LIMIT = 15
        bright_mask = np.all((phot_neb_ex_veg + 29.98) >= M_LIMIT, axis=1)
        n_before, n_after = len(phot_neb_ex_veg), np.sum(bright_mask)

        print(
            f"[Filter] Removed {n_before - n_after} clusters with M < {M_LIMIT:.1f} mag in any filter."
        )
        print(f"[Filter] Remaining clusters: {n_after} / {n_before}")

        # Only sample clusters with every band > 18 mag (apparent) - fainter than 18
        FAINT_LIMIT_APPMAG = 18
        faint_mask = np.all((phot_neb_ex_veg + args.dmod) > FAINT_LIMIT_APPMAG, axis=1)
        bright_mask = bright_mask & faint_mask
        n_after = np.sum(bright_mask)
        print(
            f"[Filter] Kept only clusters with every band > {FAINT_LIMIT_APPMAG} mag (apparent): {n_after} remaining."
        )

        # Apply mask to all relevant arrays
        cid = cid[bright_mask]
        actual_mass = actual_mass[bright_mask]
        target_mass = target_mass[bright_mask]
        form_time = form_time[bright_mask]
        eval_time = eval_time[bright_mask]
        a_v = a_v[bright_mask]
        phot_neb_ex = phot_neb_ex[bright_mask, :]
        phot_neb_ex_veg = phot_neb_ex_veg[bright_mask, :]

        filters = gal_filters[args.gal_name]

        # Sort filter names in desired order
        galful = galaxy_fullname
        filter_names = sorted(
            filters[0]
        )  # Ensures filters are ordered, e.g., f275, f336, etc.

        # Auto-search for FITS files based on sorted filter names
        fits_files = {
            filter_name: glob.glob(f"{fits_path}/{galful}/*{filter_name}*.fits")[0]
            for filter_name in filter_names
        }

        # Read FITS files dynamically based on sorted filters
        fits_data = {}
        headers = {}
        for filter_name in filter_names:
            fits_data[filter_name], headers[filter_name] = fits.getdata(
                fits_files[filter_name], header=True
            )

        # Example: Verify the order of fits_data and headers
        for filter_name in filter_names:
            print(
                f"Filter: {filter_name}, Data Shape: {fits_data[filter_name].shape}, Header: {headers[filter_name]['NAXIS1']}"
            )

        fname_list = [f.lower()[-5:] for f in filter_names]

        d = Distance(distmod=dmod * u.mag)
        distance_in_cm = d.to(u.cm).value


        if not validation:
            phot_f = phot_neb_ex / (4 * np.pi * distance_in_cm**2)
            # convert to e/s in each filter
            # Create an array of scaling factors based on PHOTFLAM values
            scaling_factors = np.array(
                [headers[filter_name]["PHOTFLAM"] for filter_name in fname_list]
            )

            # Divide each filter value in the dataset by the corresponding scaling factor
            scaled_dataset = phot_f / scaling_factors

            # convert scaled phot_neb_ex to white fluxes
            # Read FITS files
            f275 = scaled_dataset[:, 0]
            f336 = scaled_dataset[:, 1]
            f438 = scaled_dataset[:, 2]
            f555 = scaled_dataset[:, 3]
            f814 = scaled_dataset[:, 4]

            # Define function for white light image generation

            # Generate white light images
            img21ms = generate_white_light(
                [55.8, 45.3, 44.2, 65.7, 29.3], f275, f336, f438, f555, f814
            )
            img24rgb = generate_white_light(
                [0.5, 0.8, 3.1, 11.2, 13.7], f275, f336, f438, f555, f814
            )

            # Combine images for dual population
            slug_cluster_white_flux = 0.5 * (img21ms + img24rgb)
            mag_bao = np.log10(slug_cluster_white_flux) / -0.4



        # Prune padova library length to match MIST library length of 9950000
        TOTAL = int(1e6) # desired total number of clusters
        TOTAL_BLOCKED = (TOTAL // 5000) * 5000 # 50 per frame and 10 effective radii
        blocks = int(TOTAL_BLOCKED // 5000)   # number of 5000-size blocks

        # 2. 20% validation blocks
        val_blocks = round(0.2 * blocks)
        train_blocks = blocks - val_blocks
        N_VAL = val_blocks * 5000

        print(f"Number of validation clusters is {N_VAL}. \n")
        print(f"Number of training clusters is {TOTAL - N_VAL} \n")
        print(f"Ratio of validation/training = {N_VAL/(TOTAL - N_VAL)}")

        # ============================================================
        # K19 SAMPLING (EXTERNAL, FLAT-COMPATIBLE)
        # ============================================================
        if mrmodel == "Krumholz19":
            print("[K19] Sampling radii with scatter + global uniqueness...")

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            parent_idx, radii_pc = sample_k19_radii(actual_mass, n_draw=10)

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            BIN_HALF = 0.5
            N_PER_REFF = int(TOTAL_BLOCKED // len(eradius_list))   # total reff
            N_VAL_PER_REFF = int(N_VAL // len(eradius_list))       # validation number per reff

            # reff in desired order for sampling
            reff_order = [1, 10, 9, 8, 7, 6, 5, 4, 3, 2]

            used_parents = set()
            selected_by_reff = {}

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            for r in reff_order:
                mask = (radii_pc > r - BIN_HALF) & (radii_pc < r + BIN_HALF)
                candidates = parent_idx[mask]

                # remove used parents
                candidates = [pid for pid in candidates if pid not in used_parents]

                candidates = np.unique(candidates)

                print(f"[K19] reff={r}: available unique candidates = {len(candidates)}")

                if len(candidates) < N_PER_REFF:
                    raise RuntimeError(
                        f"[K19] Not enough unique clusters for reff={r}: "
                        f"{len(candidates)} < {N_PER_REFF}"
                    )

                chosen = np.array(candidates[:N_PER_REFF], dtype=int)
                selected_by_reff[r] = chosen
                used_parents.update(chosen.tolist())

            print(f"[K19] Total unique parents used = {len(used_parents)}")
            all_used = np.concatenate(list(selected_by_reff.values()))
            assert len(all_used) == len(set(all_used)), \
                "[K19] Global uniqueness violated!"



            # ------------------------------------------------------------
            # split
            # ------------------------------------------------------------
            k19_val_idx  = {}
            k19_main_idx = {}

            for r, idx in selected_by_reff.items():
                k19_val_idx[r]  = idx[:N_VAL_PER_REFF]
                k19_main_idx[r] = idx[N_VAL_PER_REFF:]

            # ------------------------------------------------------------
            # construct per-radius subsets
            # ------------------------------------------------------------
            actual_mass_val_r  = {r: actual_mass[k19_val_idx[r]] for r in eradius_list}
            eval_time_val_r    = {r: eval_time[k19_val_idx[r]]   for r in eradius_list}
            form_time_val_r    = {r: form_time[k19_val_idx[r]]   for r in eradius_list}
            a_v_val_r          = {r: a_v[k19_val_idx[r]]         for r in eradius_list}
            phot_neb_ex_val_r  = {r: phot_neb_ex[k19_val_idx[r]] for r in eradius_list}
            phot_neb_ex_veg_val_r = {r: phot_neb_ex_veg[k19_val_idx[r]] for r in eradius_list}
            mag_bao_val_r      = {r: mag_bao[k19_val_idx[r]]     for r in eradius_list}

            actual_mass_main_r = {r: actual_mass[k19_main_idx[r]] for r in eradius_list}
            eval_time_main_r   = {r: eval_time[k19_main_idx[r]]   for r in eradius_list}
            form_time_main_r   = {r: form_time[k19_main_idx[r]]   for r in eradius_list}
            a_v_main_r         = {r: a_v[k19_main_idx[r]]         for r in eradius_list}
            phot_neb_ex_main_r = {r: phot_neb_ex[k19_main_idx[r]] for r in eradius_list}
            phot_neb_ex_veg_main_r = {r: phot_neb_ex_veg[k19_main_idx[r]] for r in eradius_list}
            mag_bao_main_r     = {r: mag_bao[k19_main_idx[r]]     for r in eradius_list}





        if mrmodel == "flat":
            # Validation subset = first N_VAL clusters
            actual_mass_val = actual_mass[:N_VAL]
            eval_time_val = eval_time[:N_VAL]
            form_time_val = form_time[:N_VAL]
            a_v_val = a_v[:N_VAL]
            phot_neb_ex_val = phot_neb_ex[:N_VAL, :]
            phot_neb_ex_veg_val = phot_neb_ex_veg[:N_VAL, :]
            mag_bao_val = mag_bao[:N_VAL]

            # Training subset = remaining
            actual_mass_main = actual_mass[N_VAL:]
            eval_time_main = eval_time[N_VAL:]
            form_time_main = form_time[N_VAL:]
            a_v_main = a_v[N_VAL:]
            phot_neb_ex_main = phot_neb_ex[N_VAL:, :]
            phot_neb_ex_veg_main = phot_neb_ex_veg[N_VAL:, :]
            mag_bao_main = mag_bao[N_VAL:]

            print(
                f"[LIB] Validation subset: {N_VAL} clusters; Training subset: {TOTAL - N_VAL}"
            )
        # --- Write run summary ---
        log_dir = os.path.join(args.directory, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{args.gal_name}_{args.outname}_summary.log")

        write_summary_log(
            args,
            total_clusters=TOTAL,
            n_train=TOTAL - N_VAL,
            n_val=N_VAL,
            log_path=log_file,
            extra={
                "eradius_list": args.eradius_list,
                "nframe": args.nframe,
                "ncpus": mp.cpu_count(),
                "job_count": len(eradius_list) * (args.nframe or 0),
            },
        )
        print(f"Summary written to {log_file}")

    # Define the master seed
    master_seed = SeedSequence(int(time.time_ns()))  # Nanosecond precision
    print("Setting masterseed using nanosecond precision.")
    if not args.auto_check_missing:
        if not validation:
            nframe = val_blocks if args.make_validation_set else train_blocks
            if args.nframe is not None and args.nframe < nframe:
                nframe = args.nframe
            param_grid = list(itertools.product(eradius_list, range(nframe)))
            child_seeds = master_seed.spawn(len(param_grid))  # one seed per job
            # jobs = [(eradius, ncl, dmod, mrmodel, galaxy_fullname, validation, \
            #     overwrite, directory, galaxy_name, outname, i_frame, None, None) for eradius, i_frame in param_grid]
            jobs = [
                (
                    eradius,
                    ncl,
                    dmod,
                    mrmodel,
                    galaxy_fullname,
                    validation,
                    overwrite,
                    directory,
                    galaxy_name,
                    outname,
                    seed,
                    i_frame,
                    None,
                    None,
                    make_validation_set,
                    args.sciframe,
                    fits_path,
                    args.psf_path,
                    args.bao_path,
                    args.slug_lib_dir,
                    galaxy_names,
                    args.input_coords,
                    args.sigma_pc,
                    args.placement_mode,
                    args.placement_fits,
                    exclude_regions_parsed,
                )
                for (eradius, i_frame), seed in zip(param_grid, child_seeds)
            ]
            print(f"Running {len(jobs)} jobs for validation=False")
        else:
            framenum = args.framenum if args.framenum is not None else 0

            # Step 2: Define the parameter grid
            param_grid = list(
                itertools.product(
                    eradius_list, range(nframe_validation_start, nframe_validation_end)
                )
            )

            # Step 3: Spawn one unique seed per job
            child_seeds = master_seed.spawn(
                len(param_grid)
            )  # must come after param_grid

            #     for (eradius, i_frame_validation), seed in zip(param_grid, child_seeds)]
            jobs = [
                (
                    eradius,
                    ncl,
                    dmod,
                    mrmodel,
                    galaxy_fullname,
                    validation,
                    overwrite,
                    directory,
                    galaxy_name,
                    outname,
                    seed,
                    None,
                    i_frame_validation,
                    framenum,
                    make_validation_set,
                    args.sciframe,
                    fits_path,
                    args.psf_path,
                    args.bao_path,
                    args.slug_lib_dir,
                    galaxy_names,
                    args.input_coords,
                    args.sigma_pc,
                    args.placement_mode,
                    args.placement_fits,
                    exclude_regions_parsed,
                )
                for (eradius, i_frame_validation), seed in zip(param_grid, child_seeds)
            ]

            print("Printing jobs details. \n")
            print(jobs)
            print(f"Running {len(jobs)} jobs for validation=True")

    else:
        print("Auto check missing files")
        if args.validation:
            all_missing = []
            galx = args.gal_name
            outname = args.outname
            galaxies = np.load(os.path.join(args.directory, "galaxy_names.npy"))
            gal_filters = np.load(
                os.path.join(args.directory, "galaxy_filter_dict.npy"),
                allow_pickle=True,
            ).item()
            filts = gal_filters[galx][0]

            print("\nChecking filter white light")
            directory_syn = os.path.join(
                args.directory, args.galaxy_fullname, "white", "synthetic_fits"
            )
            print(f"Directory syn: {directory_syn}")

            expected_frames = list(
                range(nframe_validation_start, nframe_validation_end)
            )  # 0 to 99
            expected_reffs = [float(f"{x:.1f}") for x in eradius_list]  # 1.0 to 9.0

            file_pattern = f"{galx}_WFC3_UVISfF336W*frame{args.framenum}*vframe*{outname}*validation*reff*.fits"
            print(f"File pattern: {file_pattern}")
            existing_files = glob.glob(os.path.join(directory_syn, file_pattern))

            existing_entries = set()
            for file in existing_files:
                vframe_match = re.search(r"vframe_(\d+)", file)
                vframe = int(vframe_match.group(1)) if vframe_match else None

                framenum_match = re.search(r"frame(\d+)", file)
                framenum = int(framenum_match.group(1)) if framenum_match else None

                reff_match = re.search(r"reff(\d+(?:\.\d+)?)", file)
                reff = float(reff_match.group(1)) if reff_match else None

                if vframe is not None and reff is not None and framenum is not None:
                    existing_entries.add((vframe, reff, framenum))

            # Now check what's missing
            missing_files = []
            for frame in expected_frames:
                for reff in expected_reffs:
                    if (frame, reff, framenum) not in existing_entries:
                        missing_files.append(
                            (reff, frame, framenum)
                        )  # reff first for consistency

            all_missing.append(missing_files)
            print(missing_files)

            if missing_files:
                print("Missing Files:")
                for reff, frame, framen in missing_files:
                    print(
                        f"Missing: {galaxy_fullname}_white_fr{framen}_vframe{frame}_reff{reff:.2f}.fits"
                    )

                # Prepare param grid
                param_grid = missing_files
                child_seeds = master_seed.spawn(len(param_grid))
                # framenum = args.framenum if args.framenum

                # Construct jobs with seeds
                jobs = [
                    (
                        eradius,
                        ncl,
                        dmod,
                        mrmodel,
                        galaxy_fullname,
                        validation,
                        overwrite,
                        directory,
                        galaxy_name,
                        outname,
                        seed,
                        None,
                        i_frame_validation,
                        framenum,
                        make_validation_set,
                        args.sciframe,
                        fits_path,
                        args.psf_path,
                        args.bao_path,
                        args.slug_lib_dir,
                        galaxy_names,
                        args.input_coords,
                        args.sigma_pc,
                        args.placement_mode,
                        args.placement_fits,
                        exclude_regions_parsed,
                    )
                    for (eradius, i_frame_validation, framenum), seed in zip(
                        param_grid, child_seeds
                    )
                ]

            else:
                print("All expected files are present.")

    with mp.Pool(processes=min(len(jobs), mp.cpu_count())) as pool:
        pool.starmap(generate_white, jobs)
