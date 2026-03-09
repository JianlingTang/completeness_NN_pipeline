"""
Cluster library loader: SLUG library FITS. Thin wrapper; returns arrays for sampling.
"""
from pathlib import Path

import numpy as np


def load_slug_library(
    slug_lib_dir: Path,
    output_lib_dir: Path,
    allfilters_cam: list,
    mrmodel: str = "flat",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Load SLUG cluster libraries and return concatenated arrays.
    Uses the built-in slug_reader (no slugpy installation needed).

    Returns
    -------
    (cid, actual_mass, target_mass, form_time, eval_time, a_v,
     phot_neb_ex, phot_neb_ex_veg, filter_names, filter_units)
    """
    import glob
    import itertools

    from .slug_reader import read_cluster

    if mrmodel == "flat":
        libs = sorted(glob.glob(str(slug_lib_dir / "flat_in_logm_cluster_phot.fits")))
    else:
        libs0 = glob.glob(str(slug_lib_dir / "flat_in_logm_cluster_phot.fits"))
        libs1 = glob.glob(str(output_lib_dir / "*_cluster_phot.fits"))
        libs2 = glob.glob(str(slug_lib_dir / "tang*_cluster_phot.fits"))
        libs = sorted(set(itertools.chain(libs0, libs1, libs2)))
        libs = [path for path in libs if "subsolar" not in path]
    if not libs:
        raise FileNotFoundError("No SLUG library files found.")
    allfilters_cam = sorted(allfilters_cam, key=lambda x: x[-4:])
    cid_list, actual_mass_list, target_mass_list = [], [], []
    form_time_list, eval_time_list, a_v_list = [], [], []
    phot_neb_ex_list, phot_neb_ex_veg_list = [], []
    for lib in libs:
        libname = lib.split("_cluster_phot.fits")[0]
        lib_read = read_cluster(libname, read_filters=allfilters_cam, photsystem="L_lambda")
        lib_read_vega = read_cluster(libname, read_filters=allfilters_cam, photsystem="Vega")
        cid_list.append(lib_read.id)
        actual_mass_list.append(lib_read.actual_mass)
        target_mass_list.append(lib_read.target_mass)
        form_time_list.append(lib_read.form_time)
        eval_time_list.append(lib_read.time)
        a_v_list.append(lib_read.A_V)
        phot_neb_ex_list.append(lib_read.phot_neb_ex)
        phot_neb_ex_veg_list.append(lib_read_vega.phot_neb_ex)
    n = min(10_000_000_00, sum(len(c) for c in cid_list))
    cid = np.concatenate(cid_list)[:n]
    actual_mass = np.concatenate(actual_mass_list)[:n]
    target_mass = np.concatenate(target_mass_list)[:n]
    form_time = np.concatenate(form_time_list)[:n]
    eval_time = np.concatenate(eval_time_list)[:n]
    a_v = np.concatenate(a_v_list)[:n]
    phot_neb_ex = np.concatenate(phot_neb_ex_list)[:n, :]
    phot_neb_ex_veg = np.concatenate(phot_neb_ex_veg_list)[:n, :]
    filter_names = getattr(lib_read, "filter_names", [])
    filter_units = getattr(lib_read, "filter_units", [])
    return (
        cid, actual_mass, target_mass, form_time, eval_time, a_v,
        phot_neb_ex, phot_neb_ex_veg, filter_names, filter_units,
    )
