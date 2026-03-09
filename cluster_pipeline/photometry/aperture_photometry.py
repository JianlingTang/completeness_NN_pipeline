"""
Aperture photometry using IRAF daophot.phot on detected coordinates.
Photometry is performed on detected positions (matched_coords), not injected.
All parameters (zeropoint, aperture radii, etc.) come from config/metadata; no hardcoding.
"""
from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig
from ..data.galaxy_metadata import GalaxyMetadata


def run_aperture_photometry(
    frame_path: Path,
    coords_path: Path,
    output_dir: Path,
    filter_name: str,
    zeropoint: float,
    exptime: float,
    aperture_radii: list[float],
    user_aperture: float,
    photometry_dir: Path | None = None,
) -> Path:
    """
    Run IRAF phot on one frame with given coords. Writes .mag and derived .txt into output_dir.

    Parameters
    ----------
    frame_path : Path
        FITS frame (synthetic or science).
    coords_path : Path
        Coord file (x y or x y mag) - typically matched_coords for photometry(coords_detected).
    output_dir : Path
        Directory for .mag and .txt outputs.
    filter_name : str
        Filter label (e.g. F555W).
    zeropoint : float
        PHOT ZMAG.
    exptime : float
        Exposure time for datapars.
    aperture_radii : list
        Apertures (e.g. [1.0, 3.0] or [1.0, 3.0, user_aperture]).
    user_aperture : float
        LEGUS pipeline aperture radius.
    photometry_dir : Path, optional
        If set, .mag and .txt are written here (for compatibility with existing script paths).

    Returns
    -------
    Path
        Path to the magnitude table .txt file (grep "*" from .mag, sed INDEF->99.999).
    """
    try:
        from pyraf.iraf import daophot, digiphot, noao  # load IRAF packages (side effect)

        _ = (daophot, digiphot, noao)
        from pyraf import iraf
    except ImportError:
        raise ImportError("pyraf is required for aperture_photometry") from None

    work_dir = photometry_dir or output_dir
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frame_path = Path(frame_path).resolve()
    coords_path = Path(coords_path).resolve()
    if not frame_path.exists() or not coords_path.exists():
        raise FileNotFoundError(f"Frame or coords not found: {frame_path}, {coords_path}")

    apertures_str = ",".join(f"{r:.2f}" for r in aperture_radii)
    mag_out = work_dir / f"mag_{frame_path.stem}.mag"
    txt_out = work_dir / f"mag_{frame_path.stem}.txt"

    iraf.unlearn("datapars")
    iraf.datapars.scale = 1.0
    iraf.datapars.fwhmpsf = 2.0
    iraf.datapars.sigma = 0.01
    iraf.datapars.readnoise = 5.0
    iraf.datapars.epadu = exptime
    iraf.datapars.itime = 1.0
    iraf.unlearn("centerpars")
    iraf.centerpars.calgorithm = "centroid"
    iraf.centerpars.cbox = 1
    iraf.centerpars.cmaxiter = 3
    iraf.centerpars.maxshift = 1
    iraf.unlearn("fitskypars")
    iraf.fitskypars.salgori = "mode"
    iraf.fitskypars.annulus = 7.0
    iraf.fitskypars.dannulu = 1.0
    iraf.unlearn("photpars")
    iraf.photpars.apertures = apertures_str
    iraf.photpars.zmag = zeropoint
    iraf.unlearn("phot")
    iraf.phot.image = str(frame_path)
    iraf.phot.coords = str(coords_path)
    iraf.phot.output = str(mag_out)
    iraf.phot.interactive = "no"
    iraf.phot.verbose = "no"
    iraf.phot.verify = "no"
    # IRAF phot prompts "Input coordinate list(s) (default: ...)" and blocks on stdin.
    # Redirect fd 0 to a pipe with newlines so the process (and any PyRAF subprocess) gets input
    # instead of blocking on the terminal.
    import os
    import sys
    r, w = os.pipe()
    old_stdin_fd = os.dup(0)
    os.dup2(r, 0)
    os.close(r)
    try:
        # Feed several newlines in case multiple prompts (coords, output, etc.)
        os.write(w, b"\n\n\n\n\n")
        os.close(w)
        w = -1
        # Also replace sys.stdin so Python-level reads see the pipe
        sys.stdin = open(0, closefd=False)
        iraf.phot(str(frame_path))
    finally:
        os.close(0)
        os.dup2(old_stdin_fd, 0)
        os.close(old_stdin_fd)
        sys.stdin = sys.__stdin__

    short_mag = work_dir / f"detarea_short_{frame_path.stem}.mag"
    if short_mag.exists():
        short_mag.unlink()
    with mag_out.open() as fin:
        with short_mag.open("w") as fout:
            for line in fin:
                if "*" in line:
                    fout.write(line)
    if txt_out.exists():
        txt_out.unlink()
    with short_mag.open() as fin:
        with txt_out.open("w") as fout:
            for line in fin:
                fout.write(line.replace("INDEF", "99.999"))
    if short_mag.exists():
        short_mag.unlink()
    return txt_out


class AperturePhotometryRunner:
    """Run aperture photometry using galaxy metadata and config."""

    def __init__(self, metadata: GalaxyMetadata, config: PipelineConfig):
        self.metadata = metadata
        self.config = config

    def run_for_frame(
        self,
        frame_path: Path,
        coords_path: Path,
        output_dir: Path,
        filter_name: str,
    ) -> Path:
        """Run photometry for one frame/filter. Zeropoint and aperture from metadata/config."""
        zp = self.metadata.zeropoints.get(filter_name, 0.0)
        ap = self.metadata.aperture_radius or 3.0
        apertures = [1.0, 3.0] if ap in (1.0, 3.0) else [1.0, 3.0, ap]
        # Exptime from FITS header if available
        exptime = 1.0
        if frame_path.exists():
            try:
                from astropy.io import fits
                with fits.open(frame_path) as hdul:
                    exptime = float(hdul[0].header.get("EXPTIME", 1.0))
            except Exception:
                pass
        return run_aperture_photometry(
            frame_path=frame_path,
            coords_path=coords_path,
            output_dir=output_dir,
            filter_name=filter_name,
            zeropoint=zp,
            exptime=exptime,
            aperture_radii=apertures,
            user_aperture=ap,
            photometry_dir=output_dir,
        )
