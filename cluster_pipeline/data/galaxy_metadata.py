"""
Galaxy metadata: filters, zeropoints, FITS paths. Load from main_dir + galaxy_id.
Aperture radius, distance modulus, and CI cut follow perform_photometry_ci_cut_on_5filters.py
(readme when present).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read_aperture_from_readme(gal_dir: Path, gal_short: str) -> float | None:
    """Read aperture radius from readme (same pattern as perform_photometry_ci_cut_on_5filters.py)."""
    readmes = list(gal_dir.glob(f"automatic_catalog*_{gal_short}.readme"))
    if not readmes:
        return None
    try:
        content = readmes[0].read_text()
    except Exception:
        return None
    m = re.search(r"The aperture radius used for photometry is (\d+(\.\d+)?)\.", content)
    return float(m.group(1)) if m else None


def _read_dmod_and_ci_from_readme(gal_dir: Path, gal_short: str) -> tuple:
    """Read distance modulus (mag) and CI cut from readme; return (dmod, ci) or (None, None)."""
    readmes = list(gal_dir.glob(f"automatic_catalog*_{gal_short}.readme"))
    if not readmes:
        return None, None
    try:
        content = readmes[0].read_text()
    except Exception:
        return None, None
    dmod, ci = None, None
    m = re.search(
        r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)", content
    )
    if m:
        dmod = float(m.group(1))
    m = re.search(
        r"This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.", content
    )
    if m:
        ci = float(m.group(1))
    return dmod, ci


@dataclass
class GalaxyMetadata:
    """Metadata for one galaxy: filters, zeropoints, exptimes, instrument, paths to science FITS."""

    galaxy_id: str
    filters: list[str]
    zeropoints: dict[str, float]
    exptimes: dict[str, float]  # filter -> exposure time in seconds (from header_info last column)
    instrument: dict[str, str]  # filter -> instrument name
    fits_paths: dict[str, Path]  # filter -> path to drc/sci FITS
    readme_path: Path | None = None
    header_info_path: Path | None = None
    aperture_radius: float | None = None
    distance_modulus: float | None = None
    ci_cut: float | None = None

    @classmethod
    def load(cls, main_dir: Path, galaxy_id: str, galaxy_data_dir: Path | None = None) -> "GalaxyMetadata":
        """Load galaxy metadata. galaxy_filter_dict from main_dir; FITS/readme/header from galaxy_data_dir/galaxy_id or main_dir/galaxy_id."""
        gal_dir = (galaxy_data_dir if galaxy_data_dir is not None else main_dir) / galaxy_id
        gal_short = galaxy_id.split("_")[0]
        gal_filters = np.load(
            main_dir / "galaxy_filter_dict.npy", allow_pickle=True
        ).item()
        filters_list, instruments = gal_filters.get(gal_short, ([], []))
        # Sorted order = canonical 5-band wavelength order (F275W, F336W, F435W, F555W, F814W) for LEGUS
        filters = sorted(filters_list) if filters_list else []
        zp_path = gal_dir / f"header_info_{gal_short}.txt"
        zeropoints: dict[str, float] = {}
        exptimes: dict[str, float] = {}
        instrument_map: dict[str, str] = {}
        if zp_path.exists():
            # header_info: 4 columns required — filter, instrument, zeropoint, exptime (seconds)
            data = np.loadtxt(zp_path, unpack=True, skiprows=0, dtype="str")
            if data.size > 0:
                ncols = len(data) if isinstance(data, (list, tuple)) else data.shape[0]
                if ncols < 4:
                    raise ValueError(
                        f"{zp_path}: header_info must have 4 columns (filter, instrument, zeropoint, exptime). "
                        f"Found {ncols} columns. Do not use default exptime."
                    )
                filts = np.atleast_1d(data[0])
                inst = np.atleast_1d(data[1])
                zp = np.atleast_1d(data[2])
                expt = np.atleast_1d(data[3])
                for f, i, z, e in zip(filts, inst, zp, expt):
                    fkey = str(f).strip()
                    zval, eval_ = float(z), float(e)
                    for k in (fkey, fkey.upper(), fkey.lower()):
                        zeropoints[k] = zval
                        instrument_map[k] = str(i)
                        exptimes[k] = eval_
        fits_paths: dict[str, Path] = {}
        for f in filters:
            matches = list(gal_dir.glob(f"*{f}*drc.fits"))
            if not matches:
                matches = list(gal_dir.glob(f"*{f}*sci.fits"))
            if matches:
                fits_paths[f] = matches[0].resolve()
        ap = _read_aperture_from_readme(gal_dir, gal_short)
        # NGC 628c: science aperture 4 px when readme not present (LEGUS: inner ring 4 px, sky annulus at 7 px)
        if ap is None and galaxy_id == "ngc628-c":
            ap = 4.0
        dmod, ci = _read_dmod_and_ci_from_readme(gal_dir, gal_short)
        readme_path = None
        readmes = list(gal_dir.glob(f"automatic_catalog*_{gal_short}.readme"))
        if readmes:
            readme_path = readmes[0]
        return cls(
            galaxy_id=galaxy_id,
            filters=filters,
            zeropoints=zeropoints,
            exptimes=exptimes,
            instrument=instrument_map,
            fits_paths=fits_paths,
            header_info_path=zp_path if zp_path.exists() else None,
            readme_path=readme_path,
            aperture_radius=ap,
            distance_modulus=dmod,
            ci_cut=ci,
        )
