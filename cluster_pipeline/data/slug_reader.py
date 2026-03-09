"""
Pure-Python replacement for ``slugpy.read_cluster``.

Only supports FITS format (which is what SLUG library files use).
No C extensions, no GSL, no slugpy installation required — just
numpy + astropy.

Usage
-----
Drop-in replacement::

    # Before
    from slugpy import read_cluster
    data = read_cluster(libname, read_filters=filters, photsystem="Vega")

    # After
    from cluster_pipeline.data.slug_reader import read_cluster
    data = read_cluster(libname, read_filters=filters, photsystem="Vega")

The returned object is a namedtuple with the same field names as slugpy.
"""
from __future__ import annotations

import os
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy.io import fits as pyfits

# ──────────────────────────────────────────────────────────────────
# Physical constants (cgs, matching SLUG)
# ──────────────────────────────────────────────────────────────────
_c_cgs = 2.99792458e10          # speed of light  [cm/s]
_Angstrom = 1e-8                # 1 Å in cm
_pc_cm = 3.0856775814671918e18  # 1 pc in cm

# ──────────────────────────────────────────────────────────────────
# Embedded HST filter metadata
#
# pivot_wl : pivot wavelength [Å]  (from STScI instrument handbooks)
# ab_vega  : AB_mag − Vega_mag     (positive ⇒ AB is brighter number)
#
# Sources:
#   WFC3/UVIS — https://www.stsci.edu/hst/instrumentation/wfc3/
#               data-analysis/photometric-calibration
#   ACS/WFC   — https://www.stsci.edu/hst/instrumentation/acs/
#               data-analysis/zeropoints
# ──────────────────────────────────────────────────────────────────
_FILTER_META: dict[str, tuple[float, float]] = {
    # WFC3 / UVIS
    "WFC3_UVIS_F200LP": (4971.86,  0.067),
    "WFC3_UVIS_F218W":  (2228.04,  3.086),
    "WFC3_UVIS_F225W":  (2372.81,  1.532),
    "WFC3_UVIS_F275W":  (2709.69,  1.496),
    "WFC3_UVIS_F336W":  (3354.73,  1.188),
    "WFC3_UVIS_F390W":  (3923.67,  0.387),
    "WFC3_UVIS_F438W":  (4326.23, -0.095),
    "WFC3_UVIS_F475W":  (4773.10, -0.091),
    "WFC3_UVIS_F555W":  (5308.43, -0.025),
    "WFC3_UVIS_F606W":  (5889.17, -0.158),
    "WFC3_UVIS_F775W":  (7648.31, -0.389),
    "WFC3_UVIS_F814W":  (8029.30, -0.419),
    "WFC3_UVIS_F850LP": (9168.47, -0.576),
    # ACS / WFC
    "ACS_WFC_F435W":    (4328.60, -0.106),
    "ACS_WFC_F475W":    (4746.91, -0.099),
    "ACS_WFC_F555W":    (5360.95, -0.025),
    "ACS_WFC_F606W":    (5921.90, -0.160),
    "ACS_WFC_F625W":    (6311.38, -0.244),
    "ACS_WFC_F775W":    (7692.75, -0.397),
    "ACS_WFC_F814W":    (8045.00, -0.427),
    "ACS_WFC_F850LP":   (9033.13, -0.563),
    # ACS / HRC (some LEGUS fields)
    "ACS_HRC_F435W":    (4311.00, -0.104),
    "ACS_HRC_F555W":    (5356.00, -0.025),
    "ACS_HRC_F606W":    (5907.00, -0.158),
    "ACS_HRC_F814W":    (8115.00, -0.430),
}


_FILTER_ALIASES: dict[str, str] = {}
for _canon in _FILTER_META:
    _short = _canon.replace("ACS_WFC_", "ACS_").replace("ACS_HRC_", "ACS_HRC_")
    if _short != _canon and _short not in _FILTER_ALIASES:
        _FILTER_ALIASES[_short] = _canon


def _normalize_filter_name(name: str) -> str:
    """Map shorthand FITS column names (e.g. ACS_F435W) to canonical names."""
    if name in _FILTER_META:
        return name
    if name in _FILTER_ALIASES:
        return _FILTER_ALIASES[name]
    return name


def _get_filter_meta(name: str) -> tuple[float, float]:
    """Return (pivot_wl_Angstrom, AB_minus_Vega) for a filter name."""
    canon = _normalize_filter_name(name)
    if canon in _FILTER_META:
        return _FILTER_META[canon]
    raise KeyError(
        f"Filter '{name}' not in embedded lookup table. "
        f"Known filters: {sorted(_FILTER_META)}"
    )


# ──────────────────────────────────────────────────────────────────
# Internal: read cluster_prop FITS
# ──────────────────────────────────────────────────────────────────
def _read_prop_fits(prop_path: str):
    """Read ``*_cluster_prop.fits`` and return a namedtuple."""
    with pyfits.open(prop_path) as hdul:
        tbl = hdul[1].data

        cluster_id   = np.asarray(tbl.field("UniqueID"), dtype="uint64")
        trial        = np.asarray(tbl.field("Trial"),    dtype="uint64")
        time         = np.asarray(tbl.field("Time"),     dtype="float64")
        form_time    = np.asarray(tbl.field("FormTime"), dtype="float64")
        lifetime     = np.asarray(tbl.field("Lifetime"), dtype="float64")
        target_mass  = np.asarray(tbl.field("TargetMass"), dtype="float64")
        actual_mass  = np.asarray(tbl.field("BirthMass"),  dtype="float64")
        live_mass    = np.asarray(tbl.field("LiveMass"),   dtype="float64")
        try:
            stellar_mass = np.asarray(tbl.field("StellarMass"), dtype="float64")
        except KeyError:
            stellar_mass = np.full_like(live_mass, np.nan)
        num_star     = np.asarray(tbl.field("NumStar"),     dtype="uint64")
        max_star_mass = np.asarray(tbl.field("MaxStarMass"), dtype="float64")

    fields = [
        "id", "trial", "time", "form_time", "lifetime",
        "target_mass", "actual_mass", "live_mass", "stellar_mass",
        "num_star", "max_star_mass",
    ]
    values = [
        cluster_id, trial, time, form_time, lifetime,
        target_mass, actual_mass, live_mass, stellar_mass,
        num_star, max_star_mass,
    ]

    # Optional extinction columns
    with pyfits.open(prop_path) as hdul:
        col_names = [c.name for c in hdul[1].columns]
    if "A_V" in col_names:
        with pyfits.open(prop_path) as hdul:
            a_v = np.asarray(hdul[1].data.field("A_V"), dtype="float64")
        fields.append("A_V")
        values.append(a_v)
    if "A_Vneb" in col_names:
        with pyfits.open(prop_path) as hdul:
            a_vneb = np.asarray(hdul[1].data.field("A_Vneb"), dtype="float64")
        fields.append("A_Vneb")
        values.append(a_vneb)

    PropType = namedtuple("cluster_prop", fields)
    return PropType(*values)


# ──────────────────────────────────────────────────────────────────
# Internal: read cluster_phot FITS
# ──────────────────────────────────────────────────────────────────
def _read_phot_fits(
    phot_path: str,
    read_filters: Sequence[str] | None = None,
    photsystem: str | None = None,
) -> namedtuple:
    """Read ``*_cluster_phot.fits`` and return a namedtuple."""

    with pyfits.open(phot_path) as hdul:
        n_hdus = len(hdul)
        is_fits2 = n_hdus > 2

        # ── Determine filter names and units ─────────────────────
        if not is_fits2:
            # FITS1: all columns in HDU 1
            hdr = hdul[1].columns
            all_names = [c.name for c in hdr]
            all_units = [c.unit or "" for c in hdr]
            # First 3 columns: UniqueID, Trial, Time
            filter_cols = all_names[3:]
            filter_units_raw = all_units[3:]
        else:
            # FITS2: each filter in its own HDU
            filter_cols = [hdul[i].columns[0].name for i in range(2, n_hdus)]
            filter_units_raw = [hdul[i].columns[0].unit or "" for i in range(2, n_hdus)]

        # Detect _neb, _ex, _neb_ex suffixes
        base_filters: list[str] = []
        has_neb = False
        has_ex = False
        for f in filter_cols:
            if f.endswith("_neb_ex"):
                has_neb = True
                has_ex = True
            elif f.endswith("_neb"):
                has_neb = True
            elif f.endswith("_ex"):
                has_ex = True
            else:
                base_filters.append(f)

        n_variants = (1 + int(has_neb)) * (1 + int(has_ex))
        n_base = len(filter_cols) // n_variants
        base_filters = base_filters[:n_base]
        base_units = filter_units_raw[:n_base]

        # Build mapping from raw FITS column names to canonical names
        raw_to_canon = {f: _normalize_filter_name(f) for f in base_filters}
        canon_base = [raw_to_canon[f] for f in base_filters]

        # Subset to requested filters (match on canonical names)
        if read_filters is not None:
            norm_read = {_normalize_filter_name(rf) for rf in read_filters}
            fmask = [c in norm_read for c in canon_base]
            sel_filters = [c for c, m in zip(canon_base, fmask) if m]
            sel_units = [u for u, m in zip(base_units, fmask) if m]
        else:
            sel_filters = list(canon_base)
            sel_units = list(base_units)
            fmask = [True] * len(base_filters)

        nf = len(sel_filters)

        # ── Read data ────────────────────────────────────────────
        tbl = hdul[1].data
        nrows = len(tbl)
        cluster_id = np.asarray(tbl.field("UniqueID"), dtype="uint64")
        trial = np.asarray(tbl.field("Trial"), dtype="uint64")
        time = np.asarray(tbl.field("Time"), dtype="float64")

        def _read_columns(suffix: str) -> np.ndarray:
            arr = np.empty((nrows, nf), dtype="float64")
            if not is_fits2:
                for j, (bf, keep) in enumerate(zip(base_filters, fmask)):
                    if not keep:
                        continue
                    canon = raw_to_canon[bf]
                    col_idx = sel_filters.index(canon)
                    arr[:, col_idx] = tbl.field(bf + suffix)
            else:
                col_idx = 0
                for bf, keep in zip(base_filters, fmask):
                    hdu_offset = 0
                    if suffix == "_neb":
                        hdu_offset = n_base
                    elif suffix == "_ex":
                        hdu_offset = n_base * (1 + int(has_neb))
                    elif suffix == "_neb_ex":
                        hdu_offset = n_base * (1 + int(has_neb) + int(has_ex))
                    target_hdu = 2 + base_filters.index(bf) + hdu_offset
                    if keep:
                        arr[:, col_idx] = hdul[target_hdu].data.field(0)
                        col_idx += 1
            return arr

        phot = _read_columns("")
        phot_neb = _read_columns("_neb") if has_neb else None
        phot_ex = _read_columns("_ex") if has_ex else None
        phot_neb_ex = _read_columns("_neb_ex") if (has_neb and has_ex) else None

    # ── Photometric system conversion ────────────────────────────
    if photsystem is not None and photsystem != "raw":
        orig_units = list(sel_units)
        units_copy = list(orig_units)
        _convert_phot(photsystem, phot, units_copy, sel_filters)
        sel_units = units_copy
        if phot_neb is not None:
            u = list(orig_units)
            _convert_phot(photsystem, phot_neb, u, sel_filters)
        if phot_ex is not None:
            u = list(orig_units)
            _convert_phot(photsystem, phot_ex, u, sel_filters)
        if phot_neb_ex is not None:
            u = list(orig_units)
            _convert_phot(photsystem, phot_neb_ex, u, sel_filters)

    # ── Build namedtuple ─────────────────────────────────────────
    fields = ["id", "trial", "time", "filter_names", "filter_units", "phot"]
    values: list = [cluster_id, trial, time, sel_filters, sel_units, phot]
    if phot_neb is not None:
        fields.append("phot_neb")
        values.append(phot_neb)
    if phot_ex is not None:
        fields.append("phot_ex")
        values.append(phot_ex)
    if phot_neb_ex is not None:
        fields.append("phot_neb_ex")
        values.append(phot_neb_ex)

    PhotType = namedtuple("cluster_phot", fields)
    return PhotType(*values)


# ──────────────────────────────────────────────────────────────────
# Photometric conversion (L_lambda ↔ Vega / AB / L_nu)
# ──────────────────────────────────────────────────────────────────
_UNIT_MAP = {
    "L_lambda": "erg/s/A",
    "L_nu":     "erg/s/Hz",
    "AB":       "AB mag",
    "STMAG":    "ST mag",
    "Vega":     "Vega mag",
}


def _convert_phot(
    target_sys: str,
    phot: np.ndarray,
    units: list[str],
    filter_names: list[str],
) -> None:
    """In-place conversion of *phot* (N_cluster × N_filter) to *target_sys*."""
    target_unit = _UNIT_MAP.get(target_sys)
    if target_unit is None:
        raise ValueError(f"Unknown photometric system '{target_sys}'")

    nf = phot.shape[1]

    # Pre-fetch metadata
    wl_cen = np.empty(nf)
    ab_vega = np.empty(nf)
    for i, fn in enumerate(filter_names):
        pw, av = _get_filter_meta(fn)
        wl_cen[i] = pw
        ab_vega[i] = av

    for i in range(nf):
        src = units[i]
        if src == target_unit:
            continue

        col = phot[:, i]

        # First bring everything to L_nu (erg/s/Hz)
        if src == "erg/s/A":
            l_nu = col * (wl_cen[i] ** 2) * _Angstrom / _c_cgs
        elif src == "erg/s/Hz":
            l_nu = col
        elif src == "AB mag":
            f_nu = 10.0 ** (-(col + 48.6) / 2.5)
            l_nu = f_nu * 4.0 * np.pi * (10.0 * _pc_cm) ** 2
        elif src == "ST mag":
            f_lam = 10.0 ** (-(col + 21.1) / 2.5)
            l_lam = f_lam * 4.0 * np.pi * (10.0 * _pc_cm) ** 2
            l_nu = l_lam * (wl_cen[i] ** 2) * _Angstrom / _c_cgs
        elif src == "Vega mag":
            ab_mag = col + ab_vega[i]
            f_nu = 10.0 ** (-(ab_mag + 48.6) / 2.5)
            l_nu = f_nu * 4.0 * np.pi * (10.0 * _pc_cm) ** 2
        else:
            continue

        # Then convert L_nu to target
        if target_unit == "erg/s/Hz":
            phot[:, i] = l_nu
        elif target_unit == "erg/s/A":
            phot[:, i] = l_nu * _c_cgs / ((wl_cen[i] ** 2) * _Angstrom)
        elif target_unit == "AB mag":
            f_nu = l_nu / (4.0 * np.pi * (10.0 * _pc_cm) ** 2)
            phot[:, i] = -2.5 * np.log10(np.maximum(f_nu, 1e-300)) - 48.6
        elif target_unit == "ST mag":
            l_lam = l_nu * _c_cgs / ((wl_cen[i] ** 2) * _Angstrom)
            f_lam = l_lam / (4.0 * np.pi * (10.0 * _pc_cm) ** 2)
            phot[:, i] = -2.5 * np.log10(np.maximum(f_lam, 1e-300)) - 21.1
        elif target_unit == "Vega mag":
            f_nu = l_nu / (4.0 * np.pi * (10.0 * _pc_cm) ** 2)
            ab_mag = -2.5 * np.log10(np.maximum(f_nu, 1e-300)) - 48.6
            phot[:, i] = ab_mag - ab_vega[i]

        units[i] = target_unit


# ──────────────────────────────────────────────────────────────────
# Public API: read_cluster (drop-in replacement)
# ──────────────────────────────────────────────────────────────────
def read_cluster(
    model_name: str,
    output_dir: str | None = None,
    fmt: str | None = None,
    photsystem: str | None = None,
    read_filters: str | Sequence[str] | None = None,
    nofilterdata: bool = True,
    verbose: bool = False,
    **kwargs,
) -> namedtuple:
    """Read SLUG2 cluster library files and return a unified namedtuple.

    This is a drop-in replacement for ``slugpy.read_cluster`` that only
    requires ``astropy`` + ``numpy``.  It supports FITS files only
    (the standard format for distributed SLUG libraries).

    Parameters
    ----------
    model_name : str
        Base name of the SLUG model, *without* any ``_cluster_*.fits``
        suffix.  The function looks for ``{model_name}_cluster_prop.fits``
        and ``{model_name}_cluster_phot.fits``.
    output_dir : str, optional
        Directory containing the files.  If None, derived from
        *model_name* (i.e. the path part of model_name is used).
    fmt : str, optional
        Ignored (FITS only).
    photsystem : str, optional
        Target photometric system: ``"L_lambda"``, ``"L_nu"``,
        ``"AB"``, ``"STMAG"``, ``"Vega"``, or None (raw).
    read_filters : str or list[str], optional
        Subset of filters to read.
    nofilterdata : bool
        Ignored (filter metadata is embedded).
    verbose : bool
        Print progress.
    """
    if isinstance(read_filters, str):
        read_filters = [read_filters]

    # Resolve paths
    base = Path(model_name)
    if output_dir is not None:
        base = Path(output_dir) / base.name

    prop_path = str(base) + "_cluster_prop.fits"
    phot_path = str(base) + "_cluster_phot.fits"

    # ── Read prop (optional — some libs only have phot) ──────────
    prop = None
    if os.path.isfile(prop_path):
        if verbose:
            print(f"Reading cluster properties from {prop_path}")
        prop = _read_prop_fits(prop_path)

    # ── Read phot ────────────────────────────────────────────────
    if not os.path.isfile(phot_path):
        raise FileNotFoundError(f"Photometry file not found: {phot_path}")
    if verbose:
        print(f"Reading cluster photometry from {phot_path}")
    phot = _read_phot_fits(phot_path, read_filters=read_filters, photsystem=photsystem)

    # ── Merge into single namedtuple (same layout as slugpy) ─────
    out_fields = ["id", "trial", "time"]
    if prop is not None:
        out_data: list = [prop.id, prop.trial, prop.time]
    else:
        out_data = [phot.id, phot.trial, phot.time]

    if prop is not None:
        for field in prop._fields[3:]:
            out_fields.append(field)
            out_data.append(getattr(prop, field))

    for field in phot._fields[3:]:
        out_fields.append(field)
        out_data.append(getattr(phot, field))

    OutType = namedtuple("cluster_data", out_fields)
    return OutType(*out_data)
