# Read in library clusters from SLUG
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import astropy.io.fits as pyfits
import numpy as np

from pyraf import iraf

# from __future__ import print_function
# from pyraf import iraf
# from pyraf.iraf import noao, digiphot, daophot
# import astropy.io.fits as pyfits
# from astropy.io import ascii
# import logging
# import os, shutil, sys
# from scipy.ndimage import distance_transform_edt
# from astropy.io import fits
# import argparse
# import astropy.units as u
# import astropy.constants as c
# import multiprocessing as mp
# import astropy.io.fits as pyfits
# import os
# import shutil
# import sys
# import argparse
# import multiprocessing
# import glob
# import re
# import numpy as np
# from math import *
# import subprocess

def extract_core_tag(filename: str) -> str | None:
    """
    Extracts 'frameX_<outname>_reffY.fits' from filenames such as:
      - ngc628-c_acsff435w_frame0_tset_4Nov_reff3.00.fits
      - ngc628-c_acsff435w_frame12_vframe_81_vset_4Nov_validation_reff10.00.fits
      - ngc628-c_acsff435w_frame233_testset_Exp05_reff6.00.fits
      - ngc628-c_acsff435w_frame45_newset_RunX_validation_reff8.00.fits

    Returns:
      'frame0_tset_4Nov_reff3.00.fits'
      'frame12_vframe_81_vset_4Nov_validation_reff10.00.fits'
      'frame233_testset_Exp05_reff6.00.fits'
      'frame45_newset_RunX_validation_reff8.00.fits'
    """
    match = re.search(r"(frame\d{1,3}(?:_[A-Za-z0-9]+)*?_reff\d+\.\d+\.fits)", filename)
    return match.group(1) if match else None


def extract_framenum(f):
    match = re.search(r"_frame(\d+)_", f)
    return int(match.group(1)) if match else None


def extract_vframe(f):
    match = re.search(r"_vframe_(\d+)_", f)
    return int(match.group(1)) if match else None


def extract_reff(f):
    match = re.search(r"reff(\d+\.\d+)\.fits", f)
    return float(match.group(1)) if match else None


def clear_directory(directory: str, overwrite: bool = False) -> None:
    """
    Ensure the given directory exists and is empty if overwrite is True.

    Parameters:
    - directory (str): Path to the directory to clear or create.
    - overwrite (bool): If True, deletes all contents of the directory if it exists.
    """
    if os.path.exists(directory):
        if overwrite:
            shutil.rmtree(directory)
            print(f"Directory {directory} existed and was cleared.")
            os.makedirs(directory)
        else:
            print(f"Directory {directory} already exists and was not cleared.")
    else:
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    return None


def main(
    dirname: str,
    fid: int,
    dmod: float,
    sid: int,
    eid: int,
    overwrite: bool,
    outname: str,
    framenum: int,
    validation: bool,
):

    print(f"[DEBUG] dirname: {dirname}")
    print(f"[DEBUG] fid: {fid}")
    print(f"[DEBUG] dmod: {dmod}")
    print(f"[DEBUG] sid: {sid}")
    print(f"[DEBUG] eid: {eid}")
    print(f"[DEBUG] overwrite: {overwrite}")
    print(f"[DEBUG] outname: {outname}")
    print(f"[DEBUG] framenum: {framenum}")
    print(f"[DEBUG] validation: {validation}")
    main_dir = os.path.abspath(dirname or os.environ.get("COMP_MAIN_DIR", os.getcwd()))
    os.chdir(main_dir)
    # define constants
    gal_filters = np.load("galaxy_filter_dict.npy", allow_pickle=True).item()
    merr_cut = 0.3  # change to 0.5


    for i, filt in enumerate(gal_filters["ngc628-c"][0][fid : fid + 1]):
        galn, filt = dirname, filt
        gal = galn.split("_")[0]
        galdir = os.path.join(main_dir, galn)
        logfile = f"output_{gal}_{filt}_{sid}_{eid}.txt"
        # check and move the config files for SExtractor
        pydir = os.path.join(galdir, filt)
        whitedir = os.path.join(galdir, "white")
        # pipe all fits to .txt file

        fits_dir = os.path.join(pydir, "synthetic_fits")
        os.chdir(fits_dir)
        apcorrfile = f"avg_aperture_correction_{gal}.txt"

        readme_files = glob.glob(
            os.path.join(galdir, f"automatic_catalog*_{gal}.readme")
        )
        readme_file = readme_files[0]

        with open(readme_file) as f:
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
                    print(f'galdist = {float(match.group(2)) * 1e6}')
                elif "CI" in label:
                    ci = float(match.group(1))
                else:
                    useraperture = float(match.group(1))
            else:
                raise FileNotFoundError(label + " not found in the readme.")

        # import exposure time
        zpfile = os.path.join(galdir, f"header_info_{gal}.txt")
        filters, instrument, zpoint = np.loadtxt(
            zpfile, unpack=True, skiprows=0, usecols=(0, 1, 2), dtype="str"
        )
        if "5194" in gal:
            pattern = os.path.join(galdir, f"hlsp_legus_hst_*{gal}_{filt}_*sci.fits")
        else:
            pattern = os.path.join(galdir, f"hlsp_legus_hst_*{gal}_{filt}_*drc.fits")
        matching_files = glob.glob(pattern)
        if matching_files:
            sciframe = matching_files[0]
        sciframepath = os.path.abspath(sciframe)
        match = np.where(filters == filt)
        zp = float(zpoint[match])
        try:
            with pyfits.open(sciframepath) as hdulist:
                hd = hdulist[0].header
        except Exception as e:
            print("Error while reading FITS header:", str(e))
            if np.size(zp) == 0:
                sys.exit(
                    "Wrong instrument/filter names! Check the input file! \nQuitting..."
                )
        expt = hd["EXPTIME"]
        print("zp: ")
        print(zp)
        print("expt: ")
        print(expt)

        # ----------------------------------------------------------------------------------------------------------------------------------
        # STEP 3: PHOTOMETRY
        # ----------------------------------------------------------------------------------------------------------------------------------
        # creating photometry and CI folders
        if not os.path.exists(pydir + "/photometry"):
            os.makedirs(pydir + "/photometry")
        else:
            # clear_directory(pydir+'/photometry', overwrite)
            print("dir exists.")
        if not os.path.exists(pydir + "/CI"):
            os.makedirs(pydir + "/CI")
        else:
            # clear_directory(pydir+'/photometry', overwrite)
            print("dir exists.")
        #
        # Load frame and coordinate filenames
        # Load everything as a (possibly 0-D) numpy array of strings
        print("[INFO] No --eradius specified; including all reff frames.")
        frame_pattern = os.path.join(pydir, f"synthetic_fits/{gal}*{outname}*.fits")
        # coord_pattern = os.path.join(whitedir, f"synthetic_fits/{gal}_*{outname}*.coo")
        print("frame pattern is ", frame_pattern)

        if not validation:
            exclude_words = ["cmppsf", "temp", "validation", "vframe"]
            exclude_pattern = "|".join(exclude_words)
            try:
                frames_output = (
                    subprocess.check_output(
                        f"ls {frame_pattern} | grep -E -v '{exclude_pattern}'",
                        shell=True,
                        text=True,  # replaces universal_newlines=True
                    )
                    .strip()
                    .split("\n")
                )
            except subprocess.CalledProcessError:
                frames_output = []  # grep returns non-zero if no matches

            print("[DEBUG] Raw matches:", subprocess.getoutput(f"ls {frame_pattern}"))
            print("[DEBUG] Filtered frames_output:", frames_output)
        else:
            try:
                frames_output = (
                    subprocess.check_output(
                        f"ls {frame_pattern} | grep 'validation'",
                        shell=True,
                        text=True,
                    )
                    .strip()
                    .split("\n")
                )
                print("[DEBUG] Validation frames_output:", len(frames_output))
            except subprocess.CalledProcessError:
                frames_output = []
                print("[DEBUG] No validation frames found.")

        raw_frames = np.array(frames_output, dtype=str)
        # Guarantee at least 1-D, then convert to a Python list
        framename = np.atleast_1d(raw_frames).tolist()
        # coordname = atleast_1d(raw_coords).tolist()

        # Now framename and coordname are always Python lists of str
        # print("All coords:", coordname)

        # If you only want the "validation" ones:
        if validation:
            framename_validation = []
            # coordname_validation = []
            print(f'sid {sid}')
            print(f'eid {eid}')
            for i in range(sid, eid + 1):
                for f in framename:
                    if f"vframe_{i}" in f and outname in f and f"_frame{framenum}" in f:
                        framename_validation.append(f)
                    else:
                        continue
            print("length of val frames is ", len(framename_validation))
            # print("Validation coords:", coordname_validation)
        else:
            framename_validation = []
            # coordname_validation = []
            for i in range(sid, eid + 1):
                for f in framename:
                    if "validation" not in f and outname in f and f"_frame{i}_" in f:
                        framename_validation.append(f)
                # for c in coordname:
                #     if "validation" not in c and outname in c and f"_frame{i}_" in c:
                #         coordname_validation.append(c)
            print("Non-Validation frames:", framename_validation)
            # print("Validation coords:", coordname_validation)
        ######################################=- PHOTOMETRY -=#################################
        path = pydir + "/photometry"
        os.chdir(path)
        print(f'Changing to path {path}')
        print("Change path to photometry")
        print(f" length is {len(framename_validation)}")
        print(f'log file is {logfile}')
        # Redirect stdout to the output file for this worker
        with open(logfile, "w") as file:
            sys.stdout = file
            for z in range(0, len(framename_validation)):
                framepath = os.path.join(
                    pydir + "/synthetic_fits/", framename_validation[z]
                )
                os.chdir(pydir + "/s_extraction/")
                framenumber = extract_framenum(framename_validation[z])
                vframenum = extract_vframe(framename_validation[z])
                reff_values = extract_reff(framename_validation[z])
                if not validation:
                    white_coo = os.path.join(
                        whitedir,
                        f"matched_coords/matched_frame{framenumber}_{outname}_reff{reff_values:.2f}.txt",
                    )
                else:
                    white_coo = os.path.join(
                        whitedir,
                        f"matched_coords/matched_frame{framenumber}_vframe{vframenum}_{outname}_reff{reff_values:.2f}.txt",
                    )
                    print('Comparing matched.')
                if not os.path.exists(white_coo):
                    print(
                        f"[WARN] Missing white coo {white_coo}; using synthetic coords."
                    )
                    coords = os.path.join(
                        pydir,
                        "synthetic_fits",
                        framename_validation[z].replace(".fits", ".coo"),
                    )
                else:
                    coords = white_coo

                coords = white_coo
                os.chdir(path)
                iraf.unlearn("datapars")
                iraf.datapars.scale = 1.0
                iraf.datapars.fwhmpsf = 2.0
                iraf.datapars.sigma = 0.01
                iraf.datapars.readnoise = 5.0
                iraf.datapars.epadu = expt  ##  exptime
                print(expt, flush=True)
                print(type(expt), flush=True)
                iraf.datapars.itime = 1.0
                iraf.unlearn("centerpars")
                iraf.centerpars.calgorithm = (
                    "centroid"  ###centering will be done here for the whole dataset
                )
                iraf.centerpars.cbox = 1
                iraf.centerpars.cmaxiter = 3
                iraf.centerpars.maxshift = 1
                iraf.unlearn("fitskypars")
                iraf.fitskypars.salgori = "mode"
                iraf.fitskypars.annulus = 7.0
                iraf.fitskypars.dannulu = 1.0
                iraf.unlearn("photpars")
                if float(useraperture) == 3.0 or float(useraperture) == 1.0:
                    apertures = "1.0,3.0"
                else:
                    apertures = "1.0,3.0," + str(useraperture)
                iraf.photpars.apertures = apertures
                iraf.photpars.zmag = zp
                print(f"zeropoint is {zp}", flush=True)
                print(type(zp), flush=True)
                iraf.unlearn("phot")
                iraf.phot.image = framepath
                print(f"frampath input is {framepath}", flush=True)
                iraf.phot.coords = coords
                print(f"coords input is {coords}", flush=True)

                iraf.phot.output = os.path.join(
                    pydir,
                    "photometry",
                    f"mag_{os.path.basename(framename_validation[z])}.mag",
                )
                # added: remove existing .mag output if present
                if os.path.exists(iraf.phot.output):
                    os.remove(iraf.phot.output)

                print(
                    "phot output is:"
                    + os.path.join(
                        pydir,
                        "photometry",
                        f"mag_{os.path.basename(framename_validation[z])}.mag",
                    ),
                    flush=True,
                )
                iraf.phot.interactive = "no"
                iraf.phot.verbose = "no"
                iraf.phot.verify = "no"
                framename_validation[z] = os.path.basename(framename_validation[z])
                print(
                    "starting photometric analysis on frame "
                    + framename_validation[z]
                    + ".. \t using apertures: "
                    + apertures,
                    flush=True,
                )
                iraf.phot(framepath)
                # txdump detarea_f555w_3px.mag XCENTER,YCENTER,MAG,MERR,MSKY,ID mode=h > detarea_f555w_3px_short.mag
                match = re.search(r"_frame(\d+)_", framename_validation[z])
                if match:
                    framenum = int(match.group(1))
                else:
                    raise ValueError(
                        f"No frame number found in {framename_validation[z]}"
                    )
                cmd = (
                    'grep "*" mag_'
                    + framename_validation[z]
                    + f".mag > detarea_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.mag"
                )

                # added: remove existing grep target if present
                out_short = f"detarea_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.mag"
                if os.path.exists(out_short):
                    os.remove(out_short)

                os.system(cmd)
                print(f"cmd command is {cmd}", flush=True)
                cmd = (
                    f"/usr/bin/sed 's/INDEF/99.999/g' detarea_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.mag > "
                    + f"kk_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.tmp"
                )

                # added: remove existing sed target if present
                out_tmp = f"kk_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.tmp"
                if os.path.exists(out_tmp):
                    os.remove(out_tmp)
                os.system(cmd)
                source = f"kk_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.tmp"
                destination = "mag_" + framename_validation[z] + ".txt"

                # added: remove existing destination if present
                if os.path.exists(destination):
                    os.remove(destination)

                shutil.copyfile(source, destination)

                # iraf.phot.output = os.path.join(pydir + '/photometry/','mag_'+framename_validation[z]+'.mag')
                # print(f"phot output is:" + pydir + '/photometry/mag_'+framename_validation[z]+'.mag',  flush=True)
                # iraf.phot.interactive = "no"
                # iraf.phot.verbose = "no"
                # iraf.phot.verify = "no"
                # print( 'starting photometric analysis on frame '+framename_validation[z]+'.. \t using apertures: '+apertures,  flush=True)
                # iraf.phot(framepath)
                # # txdump detarea_f555w_3px.mag XCENTER,YCENTER,MAG,MERR,MSKY,ID mode=h > detarea_f555w_3px_short.mag
                # match = re.search(r'frame(\d+)', framename_validation[z])
                # if match:
                #     framenum = int(match.group(1))
                # else:
                #     raise ValueError(f"No frame number found in {framename_validation[z]}")
                # cmd = 'grep "*" mag_'+framename_validation[z]+f'.mag > detarea_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.mag'
                # os.system(cmd)
                # print(f"cmd command is {cmd}",  flush=True)
                # cmd = f"/usr/bin/sed 's/INDEF/99.999/g' detarea_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.mag > "+f"kk_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.tmp"
                # os.system(cmd)
                # source =   f'kk_{gal}_{filt}_framenum{framenum}_{framename_validation[z]}_short.tmp'
                # destination = 'mag_'+framename_validation[z]+'.txt'
                # shutil.copyfile(source, destination)

        #######################################=- CONCENTRATION INDEX CUT -=#################################
        path = pydir + "/photometry"
        os.chdir(path)

        apcorrfile = f"avg_aperture_correction_{gal}.txt"
        filepath = os.path.join(galdir, apcorrfile)

        if not os.path.exists(filepath):
            print("I cannot find the 'aperture correction' file")
            sys.exit("quitting now...")

        with open(filepath) as file:
            data = np.loadtxt(file, usecols=(0, 1, 2), dtype="str")

        filter, ap, aperr = data[:, 0], data[:, 1], data[:, 2]
        a = np.where(filter == filt)
        apcorr = ap[a]
        apcorrerr = aperr[a]
        apcorr = float(apcorr[0])
        apcorrerr = float(apcorrerr[0])
        # path = pydir + '/photometry'
        # os.chdir(path)
        for z in range(0, len(framename_validation)):
            aper, a, a, a, mag, merr = np.loadtxt(
                "mag_" + framename_validation[z] + ".txt",
                unpack=True,
                skiprows=0,
                usecols=(0, 1, 2, 3, 4, 5),
                dtype="str",
            )
            # select stars
            id = np.where(aper == "1.00")
            mag_1 = mag[id]
            id = np.where(aper == "3.00")
            mag_3 = mag[id]

            useraperture = float(useraperture)
            useraperture = f"{useraperture:.2f}"
            if float(useraperture) == 3.0:
                id = np.where(aper == "3.00")
            elif float(useraperture) == 1.0:
                id = np.where(aper == "1.00")
            else:
                id = np.where(aper == str(useraperture))
            mag_4 = mag[id]
            merr_4 = merr[id]
            mag_1 = map(float, mag_1)
            mag_3 = map(float, mag_3)
            mag_4 = map(float, mag_4)
            merr_4 = map(float, merr_4)
            mag_3 = np.asarray(list(mag_3))
            mag_1 = np.asarray(list(mag_1))
            mag_4 = np.asarray(list(mag_4))
            merr_4 = np.asarray(list(merr_4))
            values = np.subtract(mag_1, mag_3)

            # print( values)
            iraf.unlearn("txdump")
            iraf.txdump.textfiles = "mag_" + framename_validation[z] + ".mag"
            iraf.txdump.fields = "XCENTER,YCENTER"
            iraf.txdump.expr = "yes"
            tempfile = "temp" + framename_validation[z] + ".mag"
            if os.path.exists(tempfile):
                os.remove(tempfile)
            iraf.txdump(
                Stdout=tempfile,
                textfiles="mag_" + framename_validation[z] + ".mag",
                fields="XCENTER,YCENTER",
                expr="yes",
            )
            xc, yc = np.loadtxt(
                tempfile, unpack=True, skiprows=0, usecols=(0, 1), dtype="str"
            )

            # --- -- - - write the coo_*.coo file - after photometry, before ci cut
            f = open("coo_" + framename_validation[z] + ".coo", "w")
            for k in range(len(xc)):
                mag_4[k] = mag_4[k] + apcorr
                merr_4[k] = np.sqrt(merr_4[k] ** 2 + apcorrerr**2)
                if merr_cut == 0:
                    f.write(
                        xc[k]
                        + "  "
                        + yc[k]
                        + " "
                        + str(mag_4[k])
                        + " "
                        + str(merr_4[k])
                        + " "
                        + str(values[k])
                        + "\n"
                    )
                else:
                    if merr_4[k] <= merr_cut:
                        f.write(
                            xc[k]
                            + "  "
                            + yc[k]
                            + " "
                            + str(mag_4[k])
                            + " "
                            + str(merr_4[k])
                            + " "
                            + str(values[k])
                            + "\n"
                        )
            f.close()

            # --- -- - - write the ci_cut_*.coo file - after photometry, after ci cut
            if "555" in filt:
                ci = float(ci)
                id = np.where(values >= ci)
                xc = xc[id]
                yc = yc[id]
                mag_4 = mag_4[id]
                merr_4 = merr_4[id]
                ci_values = values[id]
                f = open("ci_cut_" + framename_validation[z] + ".coo", "w")
                mag_4 = np.array(list(map(float, mag_4)))
                merr_4 = np.array(list(map(float, merr_4)))
                for k in range(len(xc)):
                    if merr_cut == 0:
                        f.write(
                            xc[k]
                            + "  "
                            + yc[k]
                            + " "
                            + str(mag_4[k])
                            + " "
                            + str(merr_4[k])
                            + " "
                            + str(ci_values[k])
                            + "\n"
                        )
                    else:
                        if merr_4[k] <= merr_cut:
                            f.write(
                                xc[k]
                                + "  "
                                + yc[k]
                                + " "
                                + str(mag_4[k])
                                + " "
                                + str(merr_4[k])
                                + " "
                                + str(ci_values[k])
                                + "\n"
                            )
                f.close()
                os.system(
                    "mv ci_cut_"
                    + framename_validation[z]
                    + ".coo "
                    + pydir
                    + "/CI/ci_cut_"
                    + framename_validation[z]
                    + ".coo"
                )
                if os.path.exists("ci_cut_" + framename_validation[z] + ".coo"):
                    os.system("rm ci_cut_" + framename_validation[z] + ".coo")

        print("\n Photometry has been completed! \n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process galaxies in batches with optional overwrite."
    )
    parser.add_argument(
        "dirname", help="Name of the directory (e.g. ngc628-c_white-R17v100)"
    )
    parser.add_argument(
        "--fid", "-fid", type=int, default=0, help="Filter ID (default: 0)"
    )
    parser.add_argument("--framenum", type=int, required=False, help="Frame number 0)")
    parser.add_argument(
        "--dmod",
        "-dmod",
        type=float,
        default=29.98,
        help="Distance modulus (default: 29.98)",
    )
    parser.add_argument(
        "--sid", type=int, default=0, required=False, help="Starting frame number"
    )
    parser.add_argument(
        "--eid", type=int, default=0, required=False, help="Ending frame number"
    )
    parser.add_argument(
        "-validation",
        "--validation",
        default=False,
        action="store_true",
        help="extract validation frames?",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, clears existing output before running",
    )
    parser.add_argument("--outname", type=str, default=None, help="Output name")
    args = parser.parse_args()
    # Call main with the parsed arguments
    main(
        args.dirname,
        args.fid,
        args.dmod,
        args.sid,
        args.eid,
        args.overwrite,
        args.outname,
        args.framenum,
        args.validation,
    )
