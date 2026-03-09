import argparse
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

# Function that represents a worker process
import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.io import fits
from numpy import genfromtxt
from numpy.random import SeedSequence

# Ensure that DISPLAY is not set to avoid GUI issues
os.environ["DISPLAY"] = ""

# Path defaults: env or project root (no hardcoded absolute paths)
SCRIPT_ROOT = Path(__file__).resolve().parent.parent


def _path_env(env_key: str, default_subpath: str) -> Path:
    raw = os.environ.get(env_key)
    if raw:
        return Path(raw).resolve()
    return (SCRIPT_ROOT / default_subpath).resolve()


##############Helper functions#############################
def phys_to_pix(args: tuple[float, float, float]) -> float:
    acpx, galdist, phys = args
    theta = np.arctan(phys / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    pix_val = theta / (acpx * u.arcsec)
    return pix_val.value




def get_aperture_radius_and_metadata_from_readme(
    gal_dir: str,
    gal: str,
    eradius: float,
    dmod: float,
) -> tuple[float, float, float]:
    """
    Get useraperture (pixel), galdist (pc), and ci from readme in the same way as
    perform_photometry_ci_cut_on_5filters.py (ground truth). If readme is missing or
    a value is not found, use pipeline defaults so injection can run without readme.
    Returns (useraperture, galdist, ci).
    """
    rd_pattern = os.path.join(gal_dir, f"automatic_catalog*_{gal}.readme")
    matching_files = glob.glob(rd_pattern)
    useraperture = float(eradius)
    # galdist in pc: from dmod, d_pc = 10^((dmod+5)/5)
    galdist = 10 ** ((float(dmod) + 5) / 5.0)
    ci = 0.0
    if not matching_files:
        print(
            f"[inject_clusters] No readme found for {gal}; using eradius={useraperture}, galdist from dmod={dmod} pc, ci={ci}"
        )
        return useraperture, galdist, ci
    readme_file = matching_files[0]
    try:
        with open(readme_file) as f:
            content = f.read()
    except Exception as e:
        print(f"[inject_clusters] Could not read readme {readme_file}: {e}; using defaults.")
        return useraperture, galdist, ci
    # Same patterns as perform_photometry_ci_cut_on_5filters.py
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
                galdist = float(match.group(2)) * 1e6  # Mpc -> pc
            elif "CI" in label:
                ci = float(match.group(1))
            else:
                useraperture = float(match.group(1))
        else:
            print(f"[inject_clusters] {label} not found in readme; using default.")
    return useraperture, galdist, ci


#########################
# FITS math helper funcs
#########################
def fits_divide_scalar(in_path: str, scalar: float, out_path: str) -> None:
    """Divide FITS image by a scalar safely."""
    with fits.open(in_path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(
                f"FITS file has no image data: {in_path}. "
                "BAOlab may have failed to write the frame (check baolab log)."
            )
        data = hdul[0].data.astype(np.float32) / float(scalar)
        if data.size == 0:
            raise ValueError(
                f"FITS image is empty (size 0): {in_path}. "
                "BAOlab output may be incomplete or corrupted."
            )
        hdr = hdul[0].header
    fits.writeto(out_path, data, hdr, overwrite=True)


def fits_add_images(a_path: str, b_path: str, out_path: str) -> None:
    """Add two FITS images safely pixel-by-pixel."""
    with fits.open(a_path, memmap=False) as hdul_a, fits.open(b_path, memmap=False) as hdul_b:
        data = hdul_a[0].data.astype(np.float32) + hdul_b[0].data.astype(np.float32)
        hdr = hdul_a[0].header
    fits.writeto(out_path, data, hdr, overwrite=True)

def clear_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
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
    i_frame: int | None = None,
    i_frame_validation: int | None = None,
    framenum: int | None = None,
    filter_id: int | None = None,
    use_white: bool | None = False,
) -> int:
    # All imports are now at the top of the file for clarity and portability
    # Set up variables as before
    main_dir = os.path.abspath(os.path.normpath(directory))
    print(f"main_dir is {main_dir}")
    galaxy = galaxy_name
    galful = galaxy_fullname
    validation = validation
    print(f"validation is {validation}")
    if validation:
        print(f"i_frame_validation is {i_frame_validation}")
        print(f"framenum is {framenum}")
        print(f"filter_id is {filter_id}")
        print(f"use_white is {use_white}")
        print(f"eradius is {eradius}")
        print(f"ncl is {ncl}")
        print(f"dmod is {dmod}")
        print(f"mrmodel is {mrmodel}")
        print(f"galaxy_fullname is {galaxy_fullname}")
        print(f"galaxy_name is {galaxy_name}")
        print(f"outname is {outname}")
        print(f"i_frame is {i_frame}")
        print(f"overwrite is {overwrite}")
        print(f"directory is {directory}")
    else:
        print(f"filter_id is {filter_id}")
        print(f"use_white is {use_white}")
        print(f"eradius is {eradius}")
        print(f"ncl is {ncl}")
        print(f"dmod is {dmod}")
        print(f"mrmodel is {mrmodel}")
        print(f"galaxy_fullname is {galaxy_fullname}")
        print(f"galaxy_name is {galaxy_name}")
        print(f"outname is {outname}")
        print(f"overwrite is {overwrite}")
        print(f"directory is {directory}")

    if validation:
        i_frame_validation = i_frame_validation if i_frame_validation is not None else 0
        framenum = framenum if framenum is not None else 0
    else:
        i_frame = i_frame if i_frame is not None else 0
    dmod = dmod  # distance modulus

    filter_id = filter_id if filter_id is not None else 0
    use_white = use_white if use_white is not None else False

    fits_path = os.environ.get("COMP_FITS_PATH") or main_dir
    fits_path = os.path.abspath(fits_path)
    psf_path = os.path.join(main_dir, "PSF_files")
    if not os.path.isdir(psf_path):
        psf_path = str(_path_env("COMP_PSF_PATH", "PSF_all"))
    baopath = os.path.join(main_dir, ".deps", "local", "bin")
    bl_exe = os.path.join(baopath, "bl")
    if not os.path.isfile(bl_exe) and not os.path.isfile(bl_exe + ".exe"):
        baopath = str(_path_env("COMP_BAO_PATH", "baolab"))
    if not os.path.isdir(baopath):
        baopath = os.path.join(main_dir, ".deps", "local", "bin")
    bl_exe = os.path.join(baopath, "bl")
    if not os.path.isfile(bl_exe):
        bl_exe = os.path.join(baopath, "bl.exe")
    bl_exe = os.path.abspath(os.path.normpath(bl_exe))
    libdir = str(_path_env("COMP_OUTPUT_LIB_DIR", "output_lib"))
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
    maglim = np.array(
        [10, 26], dtype=float
    )  # CHANGE THIS TO 14, 26 TO MATCH OUR LIBRARY CUT

    # Load galaxy names and filters
    galaxy_names = [galaxy]
    galaxies = np.load(os.path.join(main_dir, "galaxy_names.npy"))
    gal_filters = np.load(
        os.path.join(main_dir, "galaxy_filter_dict.npy"), allow_pickle=True
    ).item()
    allfilters_cam = []
    cams = gal_filters["ngc628-c"]

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

    # Set up variables for photometry
    nums_perframe = ncl
    reff = eradius
    start_time = time.time()

    # Assigning variables to avoid confusion with the arguments
    if validation:
        galn, reff, i_frame, i_frame_validation, outname, filter_id = (
            galaxy_fullname,
            eradius,
            framenum,
            i_frame_validation,
            outname,
            filter_id,
        )
    else:
        galn, reff, i_frame, outname, filter_id = (
            galaxy_fullname,
            eradius,
            i_frame,
            outname,
            filter_id,
        )

    for filt in gal_filters["ngc628-c"][0][filter_id : filter_id + 1]:
        filt_orig = filt
        filt, cam = filt.lower(), cams[-1][cams[0].index(filt)]
    print(f"filt is {filt}")
    print(f"cam is {cam}")

    gal = galn.split("_")[0]
    gal_dir = os.path.join(main_dir, galn)

    pydir = os.path.join(gal_dir, filt)
    # Aperture radius and metadata: same method as perform_photometry_ci_cut_on_5filters.py; fallback when readme missing
    useraperture, galdist, ci = get_aperture_radius_and_metadata_from_readme(
        gal_dir, gal, float(eradius), dmod
    )

    # --- Science frame pattern logic ---
    if "5194" in gal:
        pattern = os.path.join(gal_dir, f"hlsp_legus_hst_*{gal}*_{filt}_*sci.fits")
    elif "white" in galn:
        if filter_id is not None:
            pattern = os.path.join(gal_dir, f"hlsp_legus_hst_*{gal}_{filt}_*drc.fits")
        else:
            pattern = os.path.join(gal_dir, "white_dualpop_s2n_white_remake.fits")
    else:
        if filter_id is not None:
            pattern = os.path.join(gal_dir, f"hlsp_legus_hst_*{gal}_{filt}_*drc.fits")
        else:
            raise FileNotFoundError("No valid filter range or galaxy pattern found.")

    matching_files = glob.glob(pattern)

    if matching_files:
        sciframe = matching_files[0]
        print(f"Found matching file: {sciframe}")
    else:
        raise FileNotFoundError(
            f"No matching file found for galaxy '{gal}' and filter '{filt}' (pattern: {pattern})"
        )
    sciframepath = os.path.abspath(os.path.normpath(sciframe))
    fname = filt  # set filter name
    cam_lower = (cam.lower() if isinstance(cam, str) else str(cam)).split("_")[0]  # e.g. WFC3_UVIS -> wfc3
    psfname = glob.glob(os.path.join(psf_path, f"psf_*_{cam_lower}_{filt}.fits"))
    if psfname:
        print(f"Found PSF file: {psfname}")
    elif "white" in galn:
        fallback_psf = os.path.join(psf_path, "psf_ngc1566_wfc3_f555w.fits")
        if os.path.isfile(fallback_psf):
            psfname = [fallback_psf]
        else:
            raise FileNotFoundError(f"PSF not found: no match for psf_*_{cam_lower}_{filt}.fits and fallback {fallback_psf} missing")
    else:
        raise FileNotFoundError(f"PSF not found: no match for psf_*_{cam_lower}_{filt}.fits in {psf_path}")
    psffile = psfname[0]
    psffilepath = psffile  # this path contains all PSF files (wfc3 and acs)

    # set bao_fhwm for photometry
    print("reff is", reff)
    print("galdist is", galdist)
    print("pixscale_wfc3 is", pixscale_wfc3)

    if cam == "wfc3":
        bao_fhwm = (reff / galdist) * (180.0 / np.pi) * (3600.0 / pixscale_wfc3) / 1.13
    elif cam == "acs":
        bao_fhwm = (reff / galdist) * (180.0 / np.pi) * (3600.0 / pixscale_acs) / 1.13
    else:
        bao_fhwm = (reff / galdist) * (180.0 / np.pi) * (3600.0 / pixscale_wfc3) / 1.13

    print(bao_fhwm)

    # set aperture correction file
    apcorrfile = f"avg_aperture_correction_{gal}.txt"

    # ---- --- -- - BAOlab zeropoint value: - -- --- ----
    baozpoint = 1e10  # change this to 1e10 for bright sources

    # ---- --- -- - scientific frame dimensions: - -- --- ----
    hd = pyfits.getheader(sciframepath)
    xaxis = hd["NAXIS1"]
    yaxis = hd["NAXIS2"]

    # ---- --- -- - finding the zeropoint and exptime - -- --- ----
    zpfile = os.path.join(gal_dir, f"header_info_{gal}.txt")
    filters, instrument, zpoint = np.loadtxt(
        zpfile, unpack=True, skiprows=0, usecols=(0, 1, 2), dtype="str"
    )
    match = np.where(filters == filt.lower())
    zp_arr = np.atleast_1d(zpoint[match])
    if zp_arr.size == 0:
        sys.exit("Wrong instrument/filter names! Check the input file! \nQuitting...")
    zp = float(zp_arr.flat[0])
    expt = hd["EXPTIME"]
    print("zp: ")
    print(zp)
    print("expt: ")
    print(expt)
    # --------------------------
    # STEP 1: SOURCE CREATION
    # --------------------------
    pydir = os.path.join(gal_dir, filt_orig)

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
    factor_zp = 10 ** (-0.4 * zp)
    factor_def = baozpoint * factor_zp

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

        cmppsf_log = f"bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt"
        test_log = f"bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt"

        with open(os.path.join(path, mk_cmppsf_bl)) as script, open(
            os.path.join(path, cmppsf_log), "w"
        ) as log:
            subprocess.run(
                [bl_exe], stdin=script, stdout=log, cwd=path, check=True
            )

        # Run mk_test_bl
        with open(os.path.join(path, mk_test_bl)) as script, open(
            os.path.join(path, test_log), "w"
        ) as log:
            subprocess.run(
                [bl_exe], stdin=script, stdout=log, cwd=path, check=True
            )

        # with open(os.path.join(path, mk_test_bl), 'r') as script, \
        #     open(os.path.join(path, test_log), 'w') as log:
        #     subprocess.run([baopath + 'bl'], stdin=script, stdout=log, cwd=path, check=True)
        print("Running baolab to create the test frames...")
        # os.system(baopath+'bl < '+mk_cmppsf_bl+f' > bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt')
        # os.system(baopath+'bl < '+mk_test_bl+f' > bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt')

        testimg1_path = os.path.join(path, testimg1)
        testimg2_path = os.path.join(path, testimg2)

        # Use full paths for both input and output
        # --- Replace test frame divisions ---
        tmp1 = testimg1_path + ".tmp"
        tmp2 = testimg2_path + ".tmp"
        fits_divide_scalar(testimg1_path, factor_def, tmp1)
        fits_divide_scalar(testimg2_path, factor_def, tmp2)
        os.replace(tmp1, testimg1_path)
        os.replace(tmp2, testimg2_path)
        # iraf.imarith(operand1=testimg1_path, op='/', operand2=factor_def, result=testimg1_path)
        # iraf.imarith(operand1=testimg2_path, op='/', operand2=factor_def, result=testimg2_path)

        # File existence check also needs full paths
        if os.path.exists(testimg1_path):
            print(f"Generated image {testimg1_path}.")
        if os.path.exists(testimg2_path):
            print(f"Generated image {testimg2_path}.")
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

        cmppsf_log = f"bao_reff{reff:.2f}_frame{i_frame}_{outname}.txt"
        test_log = f"bao_reff{reff:.2f}_frame{i_frame}_{outname}.txt"
        try:
            with open(os.path.join(path, mk_cmppsf_bl)) as script, open(
                os.path.join(path, cmppsf_log), "w"
            ) as log:
                subprocess.run(
                    [bl_exe], stdin=script, stdout=log, cwd=path, check=True
                )
        except Exception as e:
            print("Error in mk_cmppsf_bl subprocess:")
            print(locals())
            raise e

        # Run mk_test_bl with lock and debug logging
        try:
            with open(os.path.join(path, mk_test_bl)) as script, open(
                os.path.join(path, test_log), "w"
            ) as log:
                subprocess.run(
                    [bl_exe], stdin=script, stdout=log, cwd=path, check=True
                )
        except Exception as e:
            print("Error in mk_test_bl subprocess:")
            print(locals())
            raise e

        print("Running baolab to create the test frames...")
        # os.system(baopath+'bl < '+mk_cmppsf_bl+f' > bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt')
        # os.system(baopath+'bl < '+mk_test_bl+f' > bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}.txt')

        testimg1_path = os.path.join(path, testimg1)
        testimg2_path = os.path.join(path, testimg2)

        # --- Replace test frame divisions ---
        tmp1 = testimg1_path + ".tmp"
        tmp2 = testimg2_path + ".tmp"
        fits_divide_scalar(testimg1_path, factor_def, tmp1)
        fits_divide_scalar(testimg2_path, factor_def, tmp2)
        os.replace(tmp1, testimg1_path)
        os.replace(tmp2, testimg2_path)

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
    # Load FITS image
    hdul = fits.open(sciframepath)
    image_data = hdul[0].data

    # convert physical size to pixel size

    theta = np.arctan(sigma_pc * u.pc / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    if cam == "wfc3":
        pix_scale = pixscale_wfc3
    elif cam == "acs":
        pix_scale = pixscale_acs
    elif "white" in galn:
        pix_scale = pixscale_wfc3
    else:
        raise TypeError("Unknown camera type, exiting...")

    # Record the start time
    start_time = time.time()

    # --- 1) Build filenames ---
    if validation:
        mk_frames_bl = f"mk_frames_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{framenum}_vframe{i_frame_validation}_{outname}_validation.bl"
        namecoord = f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{framenum}_vframe_{i_frame_validation}_{outname}_validation.txt"
        namesource = f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{framenum}_vframe_{i_frame_validation}_{outname}_validation_temp.fits"
    else:
        mk_frames_bl = (
            f"mk_frames_{gal}_{cam}f{filt}_reff{reff:.2f}_frame{i_frame}_{outname}.bl"
        )
        namecoord = (
            f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{i_frame}_{outname}.txt"
        )
        namesource = f"{gal}_{cam}f{filt}_reff{reff:.2f}pc_sources_frame{i_frame}_{outname}_temp.fits"

    white_dir = os.path.join(gal_dir, "white")

    if use_white:
        if validation:
            if os.path.exists(
                os.path.join(
                    white_dir,
                    f"{filt}_position_{int(framenum)}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
                )
            ):
                print(
                    f"{f'{filt}_position_{int(framenum)}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt'} exists\n!"
                )
                mag_file = os.path.join(
                    white_dir,
                    f"{filt}_position_{int(framenum)}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt",
                )
                # Copy file
                shutil.copy(mag_file, os.path.join(path, namecoord))
                print(f"Copied to {namecoord}")
            else:
                print(
                    f"{f'{filt}_position_{int(framenum)}_vframe_{i_frame_validation}_{outname}_validation_reff{reff:.2f}.txt'} does not exist\n!"
                )

        else:
            if os.path.exists(
                os.path.join(
                    white_dir,
                    f"{filt_orig}_position_{int(i_frame)}_{outname}_reff{reff:.2f}.txt",
                )
            ):
                print(
                    f"{f'{filt_orig}_position_{int(i_frame)}_{outname}_reff{reff:.2f}.txt'} exists\n!"
                )
                mag_file = os.path.join(
                    white_dir,
                    f"{filt_orig}_position_{int(i_frame)}_{outname}_reff{reff:.2f}.txt",
                )
                # Copy file
                shutil.copy(mag_file, os.path.join(path, namecoord))
                print(f"Copied to {namecoord}")

    #########################################################################################
    ######### I  GENERATE THE COMPOSITE THE SYNTHETIC SOURCES FOR COMPLETENESS TEST #########
    #########################################################################################
    # Create .bl script
    mk_frames_bl_path = os.path.join(path, mk_frames_bl)
    with open(mk_frames_bl_path, "w") as f:
        f.write(
            "# I GENERATE THE TEST SOURCE IN ORDER TO GET THE REAL ZEROPOINT I NEED \n\n"
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
            + " MKSYNTH.RANDOM=NO MKSYNTH.NSTARS=100 MKSYNTH.MAGFAINT=0. MKSYNTH.MAGBRIGHT=0. MKSYNTH.MAGSTEP=0"
            + " MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT="
            + str(baozpoint)
            + " MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE="
            + cmppsf_r
            + " \n\n"
        )
        f.write("quit \n")

    print("Name source is...", namesource)
    print("Name coord is...", namecoord)
    print("Making synthetic sources on", namesource)

    # Define BAOLAB directory and output log path
    print(
        f"I am running baolab to generate the frames with synthetic sources with effective radii {reff:.2f}... \nthis could take a while... BE PATIENT!"
    )
    baolab_dir = os.path.join(pydir, "baolab")
    mk_frames_bl_path = os.path.join(baolab_dir, mk_frames_bl)

    if validation:
        bao_log_path = os.path.join(
            baolab_dir,
            f"bao_reff{reff:.2f}_frame{i_frame}_vframe{i_frame_validation}_{outname}_filter{filter_id}.txt",
        )
    else:
        bao_log_path = os.path.join(
            baolab_dir,
            f"bao_reff{reff:.2f}_frame{i_frame}_framenum{i_frame}_{outname}_filter{filter_id}.txt",
        )

    # Properly open files inside bao_lock and run subprocess
    os.environ.pop("DISPLAY", None)  # Unset DISPLAY
    with open(mk_frames_bl_path) as script, open(bao_log_path, "w") as log:
        subprocess.run(
            [bl_exe], stdin=script, stdout=log, cwd=baolab_dir, check=False
        )
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
        print(f"image {i+1} out of {len(listimg)}")

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

        print("List_image string from index 0 to -10 is", listimg[i])
        print("Final name of .fits file is ", name_final)
        # Ensure input path is full
        print(f"Current path is {path} \n")
        print(f"Confirming image name -- {listimg[i]} \n")
        print(f"Iteration = {i}")
        lism = listimg[i]
        operand1_path = os.path.join(path, lism)

        # Wait for BAOlab output to be written (avoid reading before flush)
        for _ in range(30):
            if os.path.exists(operand1_path):
                try:
                    with fits.open(operand1_path, memmap=False) as h:
                        if h[0].data is not None and getattr(h[0].data, "size", 0) > 0:
                            break
                except Exception:
                    pass
            time.sleep(0.5)
        if not os.path.exists(operand1_path):
            raise FileNotFoundError(
                f"BAOlab did not create {operand1_path} (see {bao_log_path})"
            )
        try:
            with fits.open(operand1_path, memmap=False) as h:
                if h[0].data is None or getattr(h[0].data, "size", 0) == 0:
                    raise ValueError(
                        f"BAOlab wrote empty image data: {operand1_path} (check {bao_log_path})"
                    )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Cannot read {operand1_path}: {e}") from e

        # Clean up old temp2 if it exists
        if os.path.exists(temp2):
            os.remove(temp2)

        # First imarith: divide synthetic frame
        print("Dividing .fits by sampling factor. \n")
        fits_divide_scalar(operand1_path, factor_def, temp2)
        fits_add_images(temp2, sciframepath, name_final)
        # iraf.imarith(operand1=operand1_path, op='/', operand2=factor_def, result=temp2)
        # iraf.imarith(operand1=temp2, op='+', operand2=sciframepath, result=name_final)

        # Copy to target dir (local; use shutil; scp is for remote)
        dest_dir = os.path.join(pydir, "synthetic_fits")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(name_final))
        try:
            shutil.copy2(name_final, dest_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to copy {name_final} to synthetic_fits: {e}"
            )
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
        "--nframe_start",
        type=int,
        default=0,
        help="Starting frame number for validation (int)",
    )
    parser.add_argument(
        "--nframe_end",
        type=int,
        default=1,
        help="Ending frame number for validation (int)",
    )
    parser.add_argument(
        "--framenum",
        type=int,
        default=None,
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
        "--nfilter_start", type=int, default=0, help="Starting filter number (int)"
    )
    parser.add_argument(
        "--nfilter_end", type=int, default=5, help="Ending filter number (int)"
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
        "--overwrite", action="store_true", help="Enable overwriting existing files"
    )
    parser.add_argument(
        "--use_white", action="store_true", help="Use white light catalog"
    )
    parser.add_argument(
        "--validation", action="store_true", help="Run in validation mode"
    )
    parser.add_argument(
        "--auto_check_missing", action="store_true", help="Auto check missing files"
    )
    args = parser.parse_args()

    # Define master seed
    master_seed = SeedSequence(int(time.time_ns()))
    jobs = []

    if not args.auto_check_missing:
        if not args.validation:
            print("Non-validation mode")
            nframe = range(args.nframe_start, args.nframe_end)
            nfilter = range(args.nfilter_start, args.nfilter_end)
            param_grid = list(itertools.product(args.eradius_list, nframe, nfilter))
            child_seeds = master_seed.spawn(len(param_grid))

            jobs = [
                (
                    eradius,
                    args.ncl,
                    args.dmod,
                    args.mrmodel,
                    args.galaxy_fullname,
                    args.validation,
                    args.overwrite,
                    args.directory,
                    args.gal_name,
                    args.outname,
                    seed,
                    frame_num,
                    None,
                    None,
                    filter_id,
                    args.use_white,
                )
                for (eradius, frame_num, filter_id), seed in zip(
                    param_grid, child_seeds
                )
            ]

        else:
            print("Validation mode")
            framenum = args.framenum if args.framenum is not None else 0
            nfilter = range(args.nfilter_start, args.nfilter_end)
            param_grid = list(
                itertools.product(
                    args.eradius_list,
                    range(args.nframe_validation_start, args.nframe_validation_end),
                    nfilter,
                )
            )
            child_seeds = master_seed.spawn(len(param_grid))

            jobs = [
                (
                    eradius,
                    args.ncl,
                    args.dmod,
                    args.mrmodel,
                    args.galaxy_fullname,
                    args.validation,
                    args.overwrite,
                    args.directory,
                    args.gal_name,
                    args.outname,
                    seed,
                    None,
                    i_frame_validation,
                    framenum,
                    filter_id,
                    args.use_white,
                )
                for (eradius, i_frame_validation, filter_id), seed in zip(
                    param_grid, child_seeds
                )
            ]

    else:
        print("Auto check missing mode")
        if args.validation:
            galx = args.gal_name
            outname = args.outname
            galaxies = np.load(os.path.join(args.directory, "galaxy_names.npy"))
            gal_filters = np.load(
                os.path.join(args.directory, "galaxy_filter_dict.npy"),
                allow_pickle=True,
            ).item()
            filts = gal_filters[galx][0]
            missing_files = []

            for _, filt in enumerate(filts[args.nfilter_start : args.nfilter_end]):
                fid = filts.index(filt)
                print(f"\nChecking filter: {filt}")
                directory = os.path.join(
                    args.directory, args.galaxy_fullname, filt, "synthetic_fits"
                )
                expected_frames = list(
                    range(args.nframe_validation_start, args.nframe_validation_end)
                )
                expected_reffs = [float(f"{x:.1f}") for x in args.eradius_list]

                file_pattern = (
                    f"{galx}_*{filt}*frame0*vframe*{outname}*validation*reff*.fits"
                )
                existing_files = glob.glob(os.path.join(directory, file_pattern))

                existing_entries = set()
                for file in existing_files:
                    vframe_match = re.search(r"vframe_(\d+)", file)
                    reff_match = re.search(r"reff(\d+(?:\.\d+)?)", file)
                    vframe = int(vframe_match.group(1)) if vframe_match else None
                    reff = float(reff_match.group(1)) if reff_match else None
                    if vframe is not None and reff is not None:
                        existing_entries.add((vframe, reff))

                for frame in expected_frames:
                    for reff in expected_reffs:
                        if (frame, reff) not in existing_entries:
                            missing_files.append((frame, reff, fid))
                            print((frame, reff, fid))

            if missing_files:
                child_seeds = master_seed.spawn(len(missing_files))
                framenum = args.framenum if args.framenum is not None else 0
                jobs = [
                    (
                        reff,
                        args.ncl,
                        args.dmod,
                        args.mrmodel,
                        args.galaxy_fullname,
                        args.validation,
                        args.overwrite,
                        args.directory,
                        args.gal_name,
                        args.outname,
                        seed,
                        None,
                        frame,
                        framenum,
                        fid,
                        args.use_white,
                    )
                    for (frame, reff, fid), seed in zip(missing_files, child_seeds)
                ]
            else:
                print("All expected files are present.")

    if jobs:
        print(f"Launching {len(jobs)} jobs with multiprocessing...")
        import builtins
        n_proc = builtins.min(len(jobs), mp.cpu_count())
        with mp.Pool(processes=n_proc) as pool:
            pool.starmap(main, jobs)
    else:
        print("No jobs to run.")
