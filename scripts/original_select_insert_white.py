# Read in library clusters from SLUG
import numpy as np
import os
import os.path as osp
import random
import time
import glob
from collections import namedtuple
from numpy.random import seed, rand
from slugpy import slug_pdf, read_cluster
import numpy as np
from astropy.io import fits
import os
import argparse
from numpy.random import SeedSequence, default_rng
import sys

# Function that represents a worker process
import astropy.io.fits as pyfits
import os, shutil
from astropy.io import fits
import argparse
import astropy.units as u
import astropy.constants as c
from numpy import *
from scipy.ndimage import gaussian_filter
from pyraf import iraf
import re
from scipy.ndimage import gaussian_filter
from astropy import units as u
from astropy.coordinates import *
from numpy import *
import argparse
import multiprocessing as mp
import itertools
from typing import Optional, Tuple, List, Any


##############Helper functions#############################
def phys_to_pix(args: Tuple[float, float, float]) -> float:
    acpx, galdist, phys = args
    theta = arctan(phys / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    pix_val = theta / (acpx * u.arcsec)
    return pix_val.value


def safe_imarith_div(in_path, factor):
    base, ext = os.path.splitext(in_path)
    tmp_path = base + "_tmp" + ext
    iraf.imarith(operand1=in_path, op="/", operand2=factor, result=tmp_path)
    os.replace(tmp_path, in_path)  # atomic overwrite


def safe_imarith_add_to(img1, img2, out_path):
    base, ext = os.path.splitext(out_path)
    tmp = base + "_tmp" + ext
    iraf.imarith(operand1=img1, op="+", operand2=img2, result=tmp)
    os.replace(tmp, out_path)


def mass_to_radius(args: Tuple[Any, Optional[int], str]) -> np.ndarray:
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
            sigma_MR = -0.2103855
            random_variations = np.random.randn(len(rad_lib), n_trial) * sigma_MR
            mean_R = rad_lib[:, None] + random_variations
            mean_R = np.mean(mean_R, axis=1)
        else:
            mean_R = rad_lib  # No random variations applied

    elif model == "Ryon17":
        # Compute librad using the OLS equation
        librad = -7.775 + 1.674 * log_libmass

        # Apply the cap: if log10(libmass) > 5.5, set librad to the capped value
        cap_value = -7.775 + 1.674 * 5.2  # Compute librad at log10(libmass) = 5.5
        librad = np.where(log_libmass > 5.2, cap_value, librad)

        mean_R = 10**librad  # Ensure consistency in return format

    elif model == "flat":
        # Return a random radius between 1 and 10 for each mass
        mean_R = np.log10(np.random.uniform(1, 10, size=len(libmass)))

    else:
        raise NotImplementedError("Model not implemented, exiting...")

    return mean_R


def clear_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")


def generate_white_light(
    scale_factors: List[float],
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
def main(
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
    i_frame: Optional[int] = None,
    i_frame_validation: Optional[int] = None,
    framenum: Optional[int] = None,
) -> int:
    # All imports are now at the top of the file for clarity and portability
    # Set up variables as before
    main_dir = directory
    print(f"main_dir is {main_dir}")
    os.chdir(main_dir)
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

    fits_path = "/g/data/jh2/jt4478/make_LC_copy"
    PSFpath = "/g/data/jh2/jt4478/PSF_all"
    baopath = "/g/data/jh2/jt4478/baolab-0.94.1g/"
    libdir = "/scratch/mk27/jt4478/output_lib"
    galaxies = np.load(os.path.join(main_dir, "galaxy_names.npy"))
    gal_filters = np.load(
        os.path.join(main_dir, "galaxy_filter_dict.npy"), allow_pickle=True
    ).item()
    reffs = np.array([eradius])
    nsf = False
    pixscale_wfc3 = 0.04
    pixscale_acs = 0.05
    merr_cut = 0.3
    binsize = 0.3
    nums_perframe = ncl
    sigma_pc = 100
    xcol = 0
    ycol = 1
    tolerance = 3
    minsep = False

    # Load galaxy names and filters
    galaxy_names = [galaxy]
    galaxies = np.load(os.path.join(main_dir, "galaxy_names.npy"))
    gal_filters = np.load(
        os.path.join(main_dir, "galaxy_filter_dict.npy"), allow_pickle=True
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

    # Load library clusters from SLUG library
    lib_all_list = []
    lib_all_list_veg = []
    lib_phot_files = glob.glob(os.path.join(libdir, "tang_padova*_cluster_phot.fits"))
    lib_phot_files = sorted(lib_phot_files)
    allfilters_cam.sort(
        key=lambda x: x[-4:]
    )  # Manually sort filter values to match the filter order of the LEGUS cluster catalogue (COL6-12)
    for ilib, lib in enumerate(lib_phot_files[:1]):
        libname = lib.split("_cluster_phot.fits")[0]
        print(f"Reading library clusters from file {libname}...")
        lib_read = read_cluster(
            libname, read_filters=allfilters_cam, photsystem="L_lambda"
        )
        lib_read_vega = read_cluster(
            libname, read_filters=allfilters_cam, photsystem="Vega"
        )
        phot_length = np.shape(lib_read.phot_neb_ex)
        print(f"library phot shape is {phot_length}")
        lib_all_list.append(lib_read)
        lib_all_list_veg.append(lib_read_vega)

    ncl_MIST = 9950000
    cid = []
    actual_mass = []
    form_time = []
    eval_time = []
    A_V = []
    phot_neb_ex = []
    phot_neb_ex_veg = []
    filter_names = lib_all_list[0].filter_names
    filter_units = lib_all_list[0].filter_units

    for i, lib_all in enumerate(lib_all_list):
        cid.append(lib_all.id)
        actual_mass.append(lib_all.actual_mass)
        form_time.append(lib_all.form_time)
        eval_time.append(lib_all.time)
        A_V.append(lib_all.A_V)
        phot_neb_ex.append(lib_all.phot_neb_ex)
        phot_neb_ex_veg.append(lib_all_list_veg[i].phot_neb_ex)

    cid = np.concatenate(cid)[:ncl_MIST]
    actual_mass = np.concatenate(actual_mass)[:ncl_MIST]
    form_time = np.concatenate(form_time)[:ncl_MIST]
    eval_time = np.concatenate(eval_time)[:ncl_MIST]
    A_V = np.concatenate(A_V)[:ncl_MIST]
    phot_neb_ex = np.concatenate(phot_neb_ex)[:ncl_MIST, :]
    phot_neb_ex_veg = np.concatenate(phot_neb_ex_veg)[:ncl_MIST, :]
    # Prune padova library length to match MIST library length of 9950000

    print(f"The whole library length is {np.shape(phot_neb_ex)}... \n")

    # Load galaxy names and filters
    gal_filters = np.load(
        os.path.join(fits_path, "galaxy_filter_dict.npy"), allow_pickle=True
    ).item()
    filters = gal_filters[galaxy]

    # Sort filter names in desired order
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

    phot_F = phot_neb_ex / (4 * np.pi * distance_in_cm**2)
    # convert to e/s in each filter
    # Create an array of scaling factors based on PHOTFLAM values
    scaling_factors = np.array(
        [headers[filter_name]["PHOTFLAM"] for filter_name in fname_list]
    )

    # Divide each filter value in the dataset by the corresponding scaling factor
    scaled_dataset = phot_F / scaling_factors

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
    img25ms = generate_white_light(
        [5.5, 4.3, 5.7, 9.9, 4.0], f275, f336, f438, f555, f814
    )
    img24rgb = generate_white_light(
        [0.5, 0.8, 3.1, 11.2, 13.7], f275, f336, f438, f555, f814
    )

    # Combine images for dual population
    slug_cluster_white_flux = 0.5 * (img21ms + img24rgb)
    mag_BAO = np.log10(slug_cluster_white_flux) / -0.4
    radii = mass_to_radius((actual_mass, 20, mrmodel))
    radius_mask = (10**radii > (eradius - 0.5)) & (
        10**radii < (eradius + 0.5)
    )  # radius mask to select clusters within the effective radius range using set mass-radius model

    # Apply mask to all relevant arrays
    actual_mass_filtered = actual_mass[radius_mask]
    eval_time_filtered = eval_time[radius_mask]
    form_time_filtered = form_time[radius_mask]
    A_V_filtered = A_V[radius_mask]
    mag_BAO_filtered = mag_BAO[radius_mask]
    phot_neb_ex_veg_filtered = phot_neb_ex_veg[radius_mask]

    # Sample from the filtered set using effective radii derived from the mass-radius model
    if not validation:
        valid_num = ncl  # the optimal number of sources for validation is 500 (LEGUS_CCT_start documentation) to avoild oversatuation to the existing science frame
        random_indices = np.random.randint(0, len(actual_mass_filtered), size=valid_num)
        # Select random indices from the filtered set using effective radii derived from the mass-radius model
        mass_select = actual_mass_filtered[random_indices]
        age_select = (
            eval_time_filtered[random_indices] - form_time_filtered[random_indices]
        )
        AV_select = A_V_filtered[random_indices]
        mag_BAO_select = mag_BAO_filtered[random_indices]
        mag_VEGA_select = phot_neb_ex_veg_filtered[random_indices] + dmod
        # 1) Ensure physprop directory exists under main_dir
        physprop_dir = os.path.join(main_dir, "physprop")
        os.makedirs(physprop_dir, exist_ok=True)

        # 2) Define names and corresponding arrays
        to_save = {
            f"mass_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mass_select,
            f"age_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": age_select,
            f"av_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": AV_select,
            f"mag_BAO_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mag_BAO_select,
            f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_reff{eradius}_{outname}.npy": mag_VEGA_select,
        }

        # 3) Loop and save
        for fname, arr in to_save.items():
            path = os.path.join(physprop_dir, fname)
            np.save(path, arr)
    else:
        print("Validation mode is on...")

    # Set up variables for photometry
    nums_perframe = ncl
    maglim = [mag_BAO.min(), mag_BAO.max()]
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
    gal_dir = os.path.join(main_dir, galn)
    os.chdir(gal_dir)
    pydir = os.path.join(gal_dir, "white")
    rd_pattern = os.path.join(gal_dir, f"automatic_catalog*_{gal}.readme")
    matching_files = glob.glob(rd_pattern)
    if matching_files:
        readme_file = matching_files[0]
    # try:

    with open(readme_file, "r") as f:
        content = f.read()

    # Match aperture radius, distance modulus, and CI using regular expressions
    patterns = [
        (
            r"The aperture radius used for photometry is (\d+(\.\d+)?)\.",
            "User-aperture radius",
        ),
        (
            r"Distance modulus used (\d+\.\d+) mag \((\d+\.\d+) Mpc\)",
            "Galactic distance",
        ),
        (
            r"This catalogue contains only sources with CI[ ]*>=[ ]*(\d+(\.\d+)?)\.",
            "CI value",
        ),
    ]

    for pattern, label in patterns:
        match = re.search(pattern, content)
        if match:
            if "distance" in label:
                galdist = float(match.group(2)) * 1e6
            elif "CI" in label:
                ci = float(match.group(1))
            else:
                useraperture = float(match.group(1))
        else:
            raise FileNotFoundError(label + " not found in the readme.")

    # set science frame name and path
    if "5194" in gal:
        pattern = os.path.join(gal_dir, f"hlsp_legus_hst_*{gal}*_{filt}_*sci.fits")
    elif "white" in galn:
        pattern = os.path.join(gal_dir, "white_dualpop_s2n_white_remake.fits")
        # pattern = os.path.join(gal_dir, f"hlsp_legus_hst_*{gal}_{filt}_*drc.fits")
    else:
        raise FileNotFoundError("Exiting...")

    matching_files = glob.glob(pattern)

    if matching_files:
        sciframe = matching_files[0]
        print(f"Found matching file: {sciframe}")
    else:
        print(f"No matching file found for galaxy '{gal}' and filter '{filt}'")
    sciframepath = os.path.join(gal_dir, sciframe)
    fname = filt  # set filter name
    psfname = glob.glob(os.path.join(PSFpath, f"psf_*_{cam}_{filt}.fits"))
    if psfname:
        print(f"Found PSF file: {psfname}")
    elif "white" in galn:
        psfname = [os.path.join(PSFpath, "psf_ngc1566_wfc3_f555w.fits")]
    else:
        raise FileNotFoundError
    psffile = psfname[0]
    psffilepath = psffile  # this path contains all PSF files (wfc3 and acs)

    # set baoFHWM for photometry
    print("reff is", reff)
    print("galdist is", galdist)
    print("pixscale_wfc3 is", pixscale_wfc3)

    if cam == "wfc3":
        baoFHWM = (reff / galdist) * (180.0 / pi) * (3600.0 / pixscale_wfc3) / 1.13
    elif cam == "acs":
        baoFHWM = (reff / galdist) * (180.0 / pi) * (3600.0 / pixscale_acs) / 1.13
    else:
        baoFHWM = (reff / galdist) * (180.0 / pi) * (3600.0 / pixscale_wfc3) / 1.13

    print(baoFHWM)

    # set aperture correction file
    apcorrfile = f"avg_aperture_correction_{gal}.txt"

    # ---- --- -- - BAOlab zeropoint value: - -- --- ----
    baozpoint = 1e3
    #     baozpoint = 1 ##### CHANGE OF ZP for white light!

    # ---- --- -- - scientific frame dimensions: - -- --- ----
    hd = pyfits.getheader(sciframe)
    xaxis = hd["NAXIS1"]
    yaxis = hd["NAXIS2"]
    # coordinates for test source:
    # xcss=int(xaxis/2)
    # ycss=int(yaxis/2)

    # ---- --- -- - finding the zeropoint and exptime - -- --- ----
    zpfile = os.path.join(gal_dir, f"header_info_{gal}.txt")
    filters, instrument, zpoint = loadtxt(
        zpfile, unpack=True, skiprows=0, usecols=(0, 1, 2), dtype="str"
    )
    match = where(filters == fname)
    # Set zp to be 1 for easy computation
    zp = 1
    expt = hd["EXPTIME"]
    print("zp: ")
    print(zp)
    print("expt: ")
    print(expt)
    # --------------------------
    # STEP 1: SOURCE CREATION
    # --------------------------

    pydir = os.path.join(gal_dir, "white")

    # Clear or create the baolab directory
    # if i_frame < 1:
    clear_directory(os.path.join(pydir, "baolab"))

    # Clear or create the synthetic_frames directory
    clear_directory(os.path.join(pydir, "synthetic_frames"))

    # Clear or create the synthetic_fits directory
    clear_directory(os.path.join(pydir, "synthetic_fits"))

    #########################################################################################
    ######### I  GENERATE 2 TEST SOURCES FOR A VISUAL CHECK OF THE SIMULATED SOURCES ########
    #########################################################################################
    path = pydir + "/baolab"
    os.chdir(path)
    # factor_zp=10**(-0.4*zp)
    factor_def = baozpoint

    # name a unique .bl file to avoid multi-core overstepping.
    if validation:
        mk_cmppsf_bl = f"mk_cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.bl"
        cmppsf_r0 = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}_test{reffs[0]:.2f}pc.fits"
        cmppsf_r1 = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}_test{reffs[-1]:.2f}pc.fits"
        cmppsf_r = f"cmppsf_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.fits"

        f = open(mk_cmppsf_bl, "w")
        f.write("# I GENERATE THE COMPOSITE PSFs \n\n")
        f.write(
            "mkcmppsf "
            + cmppsf_r
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(baoFHWM)
            + " MKCMPPSF.FWHMOBJY="
            + str(baoFHWM)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r0
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(baoFHWM)
            + " MKCMPPSF.FWHMOBJY="
            + str(baoFHWM)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r1
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(baoFHWM)
            + " MKCMPPSF.FWHMOBJY="
            + str(baoFHWM)
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
        g1 = open(testcoord1, "w")
        g1.write("50 50 " + str(maglim[0]))
        g1.close()

        testcoord2 = (
            f"testcoord2_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}"
            + ".txt"
        )
        g2 = open(testcoord2, "w")
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
        f = open(mk_test_bl, "w")
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
        os.system(
            baopath
            + "bl < "
            + mk_cmppsf_bl
            + f" > bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt"
        )
        os.system(
            baopath
            + "bl < "
            + mk_test_bl
            + f" > bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt"
        )

        # iraf.imarith(operand1 = testimg1, op = '/', operand2 = factor_def, result = testimg1)
        # iraf.imarith(operand1 = testimg2, op = '/', operand2 = factor_def, result = testimg2)
        safe_imarith_div(testimg1, factor_def)
        safe_imarith_div(testimg2, factor_def)

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

        f = open(mk_cmppsf_bl, "w")
        f.write("# I GENERATE THE COMPOSITE PSFs \n\n")
        f.write(
            "mkcmppsf "
            + cmppsf_r
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(baoFHWM)
            + " MKCMPPSF.FWHMOBJY="
            + str(baoFHWM)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r0
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(baoFHWM)
            + " MKCMPPSF.FWHMOBJY="
            + str(baoFHWM)
            + " MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE="
            + psffilepath
            + " \n\n"
        )
        f.write(
            "mkcmppsf "
            + cmppsf_r1
            + " MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX="
            + str(baoFHWM)
            + " MKCMPPSF.FWHMOBJY="
            + str(baoFHWM)
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
        g1 = open(testcoord1, "w")
        g1.write("50 50 " + str(maglim[0]))
        g1.close()

        testcoord2 = (
            f"testcoord2_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}"
            + ".txt"
        )
        g2 = open(testcoord2, "w")
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
        f = open(mk_test_bl, "w")
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
        os.system(
            baopath
            + "bl < "
            + mk_cmppsf_bl
            + f" > bao_reff{reff:.2f}_frame{i_frame}_{outname}.txt"
        )
        os.system(
            baopath
            + "bl < "
            + mk_test_bl
            + f" > bao_reff{reff:.2f}_frame{i_frame}_{outname}.txt"
        )

        # iraf.imarith(operand1 = testimg1, op = '/', operand2 = factor_def, result = testimg1)
        # iraf.imarith(operand1 = testimg2, op = '/', operand2 = factor_def, result = testimg2)
        safe_imarith_div(testimg1, factor_def)
        safe_imarith_div(testimg2, factor_def)

        if os.path.exists(testimg1):
            print(f"Generated image {testimg1}.")
        if os.path.exists(testimg2):
            print(f"Generated image {testimg2}.")

    #########################################################################################
    ######### I  GENERATE THE COMPOSITE THE SYNTHETIC SOURCES FOR COMPLETENESS TEST #########
    #########################################################################################
    print(
        "I am writing the baolab file to generate the frames with synthetic sources..."
    )
    # Load FITS image
    hdul = fits.open(sciframe)
    image_data = hdul[0].data
    zero_mask = image_data == 0
    image_data_flat = image_data.flatten()

    # convert physical size to pixel size

    theta = arctan(sigma_pc * u.pc / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    if cam == "wfc3":
        pix_scale = pixscale_wfc3
    elif cam == "acs":
        pix_scale = pixscale_acs
    elif "white" in galn:
        pix_scale = pixscale_wfc3
    else:
        raise TypeError("Unknown camera type, exiting..")

    if nsf:
        filtered = image_data
        image_data_flat = image_data.flatten()
        # Get indices where values are greater than 0
        positive_indices_image = where(image_data_flat == 0)

        # Create boolean arrays with the same shape as the original arrays
        image_data_positive = zeros_like(image_data_flat, dtype=bool)

        # Mark the positive indices as True
        image_data_positive[positive_indices_image] = True

        # Element-wise logical AND operation
        p_id = image_data_positive
        positive_indices_result = where(p_id)[0]
        image_shape = image_data.shape
        #     zero_mask = image_data_flat == 0

        # Now combine the safe indices with the original positive_indices_result
        safe_positive_indices_result = positive_indices_result

    else:
        # convert to pixel scale and gaussian-convolve with sigma = pixel scale
        pix_cl = (sigma_pc / (2 * pi * galdist) * 3600) / pix_scale
        # convert pc scale to pixel-scale for gaussian_filter function
        filtered = gaussian_filter(image_data, sigma=pix_cl)
        filtered_flat = filtered.flatten()

        # Get indices where values are greater than 0
        positive_indices_image = where(image_data_flat > 0)
        positive_indices_filtered = where(filtered_flat > 0)

        # Create boolean arrays with the same shape as the original arrays
        image_data_positive = zeros_like(image_data_flat, dtype=bool)
        filtered_positive = zeros_like(filtered_flat, dtype=bool)

        # Mark the positive indices as True
        image_data_positive[positive_indices_image] = True
        filtered_positive[positive_indices_filtered] = True

        # Element-wise logical AND operation
        p_id = logical_and(image_data_positive, filtered_positive)
        positive_indices_result = where(p_id)[0]
        image_shape = image_data.shape

    theta = arctan(
        reff * 3 * u.pc / (galdist * u.pc)
    )  # select minimum separation value to be 3 times of the effective radius
    theta = theta.to(u.arcsec)
    pix_cl = theta / (pix_scale * u.arcsec)
    min_separation = 3 * pix_cl.value

    # Record the start time
    start_time = time.time()

    # Initialize the list to store selected coordinates
    selected_coordinates = []
    rng = default_rng(seed)  # seed is a SeedSequence object passed from main()

    while len(selected_coordinates) < nums_perframe:
        # Generate a random coordinate
        random_coord = rng.choice(positive_indices_result)

        # Convert the random coordinate to (x, y)
        x, y = unravel_index(random_coord, image_shape)

        if (y, x) in selected_coordinates:
            continue

        # Flag to check if the coordinate meets the minimum separation requirement
        if minsep:
            meets_requirement = True

            if len(selected_coordinates) > 0:
                # Check the distance to the nearest neighbor for each selected coordinate
                x_sel, y_sel = array(
                    [coord[0] for coord in selected_coordinates]
                ), array([coord[1] for coord in selected_coordinates])
                distance = np.sqrt((x - x_sel) ** 2 + (y - y_sel) ** 2)
                if np.any(distance < min_separation):
                    meets_requirement = False

            if not meets_requirement:
                continue  # Restart the loop for a new random coordinate
            else:
                selected_coordinates.append((y, x))
        else:
            selected_coordinates.append((y, x))

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
        BAO_mag_select = mag_BAO_select
        mag_VEGA_select = mag_VEGA_select

        # 2a) Filter-specific magnitude files
        for ifn, fname in enumerate(filter_names):
            fn = os.path.join(
                pydir, f"{fname}_position_{i_frame}_{outname}_reff{reff:.2f}.txt"
            )
            if not os.path.exists(fn):
                with open(fn, "w") as fh:
                    for (y, x), mag in zip(
                        selected_coordinates, mag_VEGA_select[:, ifn]
                    ):
                        fh.write(f"{y} {x} {mag}\n")
            else:
                print(f"Filter mag file exists: {fn}")

        # 2b) White + coord file
        white_fn = os.path.join(
            pydir, f"white_position_{i_frame}_{outname}_reff{reff:.2f}.txt"
        )
        if os.path.exists(white_fn):
            print(f"White file exists; copying to {namecoord}")
            shutil.copy(white_fn, namecoord)
        else:
            print("Creating white + coord files.")
            with open(white_fn, "w") as wf, open(namecoord, "w") as nc:
                for (y, x), mag in zip(selected_coordinates, BAO_mag_select):
                    wf.write(f"{y} {x} {mag}\n")
                    nc.write(f"{y} {x} {mag}\n")

    # --- 3) Validation mode: reload mags and write validation files ---
    else:
        # 3a) Load BAO mags from existing white file
        white_orig = os.path.join(
            pydir, f"white_position_{i_frame}_flat_r10_reff{reff:.2f}.txt"
        )  # TESTING
        print(f"TEMPORARY WHITE FILE: {white_orig}")
        if not os.path.exists(white_orig):
            raise FileNotFoundError(f"Missing validation white file: {white_orig}")

        BAO_mag_select = np.loadtxt(white_orig)[:, -1]
        mag_VEGA_select = np.zeros((len(BAO_mag_select), len(filter_names)))

        # 3b) Rebuild mag_VEGA_select from filter files
        for ifn, fname in enumerate(filter_names):
            fn = os.path.join(
                pydir, f"{fname}_position_{i_frame}_flat_r10_reff{reff:.2f}.txt"
            )
            mag_VEGA_select[:, ifn] = np.loadtxt(fn)[:, -1]

        # 3c) Write validation filter files
        for ifn, fname in enumerate(filter_names):
            fn_val = os.path.join(
                pydir,
                f"{fname}_position_{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
            )
            if not os.path.exists(fn_val):
                with open(fn_val, "w") as fh:
                    for (y, x), mag in zip(
                        selected_coordinates, mag_VEGA_select[:, ifn]
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
            shutil.copy(white_val, namecoord)
        else:
            print(f"Creating validation white + coord files: {white_val}")
            # sys.exit('exiting.')
            with open(white_val, "w") as wf, open(namecoord, "w") as nc:
                for (y, x), mag in zip(selected_coordinates, BAO_mag_select):
                    wf.write(f"{y} {x} {mag}\n")
                    nc.write(f"{y} {x} {mag}\n")

    #########################################################################################
    ######### I  GENERATE THE COMPOSITE THE SYNTHETIC SOURCES FOR COMPLETENESS TEST #########
    #########################################################################################

    f = open(mk_frames_bl, "w")
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

    print(
        f"I am running baolab to generate the frames with synthetic sources with effective radii{reff:.2f}... \nthis could take a while... BE PATIENT!"
    )
    # print(baopath+" bl < mk_frames_"+"{:.2f}".format(reff)+str(i_frame)+"pc.bl > bao.txt")
    os.system(baopath + "bl < " + mk_frames_bl + f" > bao_{i_frame}_{outname}.txt")
    os.system("scp *pc_sources_*.txt " + pydir + "/synthetic_fits/")
    ##########################################################################################
    ########### I  DIVIDE THE FRAMES FOR THE CORRECT FACTOR AND ADD THE BACKGROUND ###########
    ##########################################################################################
    print("")
    print("I am adding the newly generated synthetic frames to the background...")
    print(
        "The resulting frames of this operation will be the ones to use for completeness test!"
    )
    print("But this operation can take several minutes! BE PATIENT (again!!!)...")
    if not validation:
        os.system(
            "ls "
            + namesource
            + f" > list_temp_{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}.txt"
        )  # pipe these fits files to a txt file
        listimg = genfromtxt(
            f"list_temp_{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}.txt",
            dtype="str",
        )
    else:
        os.system(
            "ls "
            + namesource
            + f" > list_temp_{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt"
        )  # pipe these fits files to a txt file
        listimg = genfromtxt(
            f"list_temp_{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
            dtype="str",
        )
    if size(listimg) == 1:
        listimg = append(listimg, "string")
        listimg = listimg[where(listimg[1] == "string")]
        print("size listimg is 1")
    else:
        print("size listimg is not 1...")
    for i in range(0, len(listimg)):
        print("image " + str(i + 1) + " out of " + str(len(listimg)))
        if not validation:
            name_final = (
                f"{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}.fits"
            )
        else:
            name_final = f"{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.fits"
        print("list_image string from index 0 to -10 is", listimg[i])
        print("final name is", name_final)
        if not validation:
            temp2 = (
                f"{gal}_{cam}f{filt}_frame{i_frame}_{outname}_reff{reff:.2f}_temp2.fits"
            )
        else:
            temp2 = f"{gal}_{cam}f{filt}_frame{i_frame}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}_temp2.fits"
        # iraf.imarith(operand1=listimg[i],   op ='/',   operand2=factor_def,   result=temp2)
        # iraf.imarith(operand1=temp2,   op='+',   operand2=sciframepath,   result=name_final)
        safe_imarith_div_to(listimg[i], factor_def, temp2)
        safe_imarith_add_to(temp2, sciframepath, name_final)
        try:
            os.system("scp " + name_final + " " + pydir + "/synthetic_fits/")
        except:
            raise FileNotFoundError(f"{name_final} is not found in /synthetic_fits/...")

    ######################################################################################################
    ################ MAKE SOME ORDER BEFORE STARTING WITH EXTRACTION AND PHOTOMETRY !!!!! ################
    ######################################################################################################

    # move synthetic frames
    os.chdir(pydir + "/baolab/")
    path = pydir + "/synthetic_fits/"
    os.chdir(path)
    os.chdir(pydir)

    print("\n Source creation is completed! \n")

    # End time
    end_time = time.time()

    # Compute elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process galaxies in batches.")
    parser.add_argument(
        "--gal_name", type=str, default=None, help="name of the galaxy (str)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="main directory of the completeness calculation (str)",
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
        "--framenum",
        type=int,
        default=0,
        help="Frame number to perform validation on (int, optional)",
    )
    parser.add_argument(
        "--galaxy_fullname",
        type=str,
        default="ngc628-c_white-R17v100",
        help="full name of the galaxy (str)",
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
        "--overwrite",
        default=False,
        action="store_true",
        help="Enable overwriting existing files",
    )
    parser.add_argument(
        "--use_white",
        default=False,
        action="store_true",
        help="Use white light catalog",
    )
    parser.add_argument(
        "--validation",
        default=False,
        action="store_true",
        help="Run in validation mode",
    )
    parser.add_argument(
        "--auto_check_missing", action="store_true", help="Auto check missing files"
    )
    args = parser.parse_args()

    eradius_list = args.eradius_list
    ncl = args.ncl
    dmod = args.dmod
    mrmodel = args.mrmodel
    directory = args.directory
    galaxy_fullname = args.galaxy_fullname
    galaxy_name = args.gal_name
    outname = args.outname
    validation = args.validation
    overwrite = args.overwrite
    nframe_validation_start = args.nframe_validation_start
    nframe_validation_end = args.nframe_validation_end
    auto_check_missing = args.auto_check_missing

    # Define the master seed
    master_seed = SeedSequence(int(time.time_ns()))  # Nanosecond precision

    if not args.auto_check_missing:
        if not validation:
            nframe = args.nframe if args.nframe is not None else 100
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
                    i_frame,
                    None,
                    None,
                    seed,
                )
                for (eradius, i_frame), seed in zip(param_grid, child_seeds)
            ]
            print(f"Running {len(jobs)} jobs for validation=False")
        else:
            # framenum = args.framenum if args.framenum is not None else 0
            # child_seeds = master_seed.spawn(len(param_grid))
            # param_grid = list(itertools.product(eradius_list, range(nframe_validation_start, nframe_validation_end)))
            # jobs = [(eradius, ncl, dmod, mrmodel, galaxy_fullname, validation,\
            #      overwrite, directory, galaxy_name, outname, None, i_frame_validation, framenum) for eradius, i_frame_validation in param_grid]
            # Step 1: Set default value
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

            # Step 4: Construct jobs including seed
            # jobs = [
            #     (
            #         eradius, ncl, dmod, mrmodel, galaxy_fullname, validation,
            #         overwrite, directory, galaxy_name, outname,
            #         None, i_frame_validation, framenum, seed
            #     )
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
            galaxies = np.load("/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_names.npy")
            gal_filters = np.load(
                "/g/data/jh2/jt4478/make_LEGUS_CCT/galaxy_filter_dict.npy",
                allow_pickle=True,
            ).item()
            filts = gal_filters[galx][0]

            print(f"\nChecking filtet white light")
            directory_syn = os.path.join(
                args.directory, args.galaxy_fullname, "white", "synthetic_fits"
            )
            expected_frames = list(range(100))  # 0 to 99
            expected_reffs = [float(f"{x:.1f}") for x in range(1, 10)]  # 1.0 to 9.0

            file_pattern = f"ngc628-c_WFC3_UVISfF336W*frame0*vframe*{outname}*validation*reff*.fits"
            existing_files = glob.glob(os.path.join(directory_syn, file_pattern))

            existing_entries = set()
            missing_files = []

            for file in existing_files:
                vframe_match = re.search(r"vframe_(\d+)", file)
                vframe = int(vframe_match.group(1)) if vframe_match else None

                reff_match = re.search(r"reff(\d+(?:\.\d+)?)", file)
                reff = float(reff_match.group(1)) if reff_match else None

                if vframe is not None and reff is not None:
                    existing_entries.add((vframe, reff))

            for frame in expected_frames:
                for reff in expected_reffs:
                    if (frame, reff) not in existing_entries:
                        missing_files.append((frame, reff))

            all_missing.append(missing_files)
            print(missing_files)

            if missing_files:
                print("Missing Files:")
                for (
                    frame,
                    reff,
                ) in missing_files:
                    print(f"Missing: ngc628-c_white_vframe{frame}_reff{reff:.2f}.fits")
                param_grid = [(reff, frame) for frame, reff in missing_files]
                child_seeds = master_seed.spawn(len(param_grid))
                framenum = args.framenum if args.framenum is not None else 0
                jobs = [
                    (
                        eradius,
                        ncl,
                        dmod,
                        mrmodel,
                        galaxy_fullname,
                        validation,  # validation = True
                        overwrite,
                        directory,
                        galaxy_name,
                        outname,
                        None,
                        i_frame_validation,
                        framenum,
                        seed,
                    )
                    for (eradius, i_frame_validation), seed in zip(
                        param_grid, child_seeds
                    )
                ]
                print(f"Running {len(jobs)} auto-check jobs")
            else:
                print("All expected files are present.")
            # jobs = []
            # for frame, reff in missing_files:
            #     eradius = reff
            #     i_frame_validation = frame
            #     framenum = args.framenum if args.framenum is not None else 0

            #     jobs.append((eradius, ncl, dmod, mrmodel, galaxy_fullname, validation, \
            #                  overwrite, directory, galaxy_name, outname, None, i_frame_validation, framenum))
            # print(len(jobs))

    with mp.Pool(processes=min(len(jobs), mp.cpu_count())) as pool:
        pool.starmap(main, jobs)
