#!/usr/bin/env python3
"""
Smoke test: verify IRAF + PyRAF aperture photometry (same path as the pipeline).

Run on Gadi login node (or anywhere with IRAF + pyraf):

  cd /path/to/comp_pipeline_restructure
  export IRAF=/path/to/iraf
  source .venv/bin/activate
  python scripts/test_photometry_iraf_smoke.py

Success: prints paths to .mag / .txt and exits 0.
Failure: ImportError (missing pyraf) → exit 2; photometry error → exit 1.

Optional:
  python scripts/test_photometry_iraf_smoke.py --keep /tmp/phot_smoke_out
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_synthetic_fits(path: Path, shape: tuple[int, int] = (256, 256)) -> None:
    """Small FITS with a Gaussian blob + sky; EXPTIME in header for datapars."""
    from astropy.io import fits

    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    cx, cy = nx / 2.0, ny / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    data = 2000.0 * np.exp(-r2 / (2.0 * 4.0**2)) + 15.0
    data = data.astype(np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["EXPTIME"] = (500.0, "exposure seconds")
    hdu.writeto(path, overwrite=True)


def _write_coords(path: Path, xy: list[tuple[float, float]]) -> None:
    with path.open("w") as f:
        for x, y in xy:
            f.write(f"{x:.2f} {y:.2f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="IRAF/PyRAF photometry smoke test for login node.")
    ap.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="If set, use this directory and do not delete outputs (for debugging).",
    )
    args = ap.parse_args()

    try:
        from cluster_pipeline.photometry.aperture_photometry import run_aperture_photometry
    except ImportError as e:
        print("ERROR: cannot import aperture_photometry:", e, file=sys.stderr)
        return 2

    if args.keep:
        work = Path(args.keep).resolve()
        work.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="phot_smoke_")
        work = Path(tmp_ctx.name)

    try:
        frame = work / "smoke_frame.fits"
        coords = work / "smoke_coords.coo"
        out_dir = work / "out"

        _write_synthetic_fits(frame)
        # Pixel coords on the synthetic image (center + offset)
        _write_coords(coords, [(128.0, 128.0), (96.0, 96.0)])

        print("==> Synthetic FITS:", frame)
        print("==> Coords:", coords)
        print("==> Running run_aperture_photometry (IRAF phot)...")

        txt_path = run_aperture_photometry(
            frame_path=frame,
            coords_path=coords,
            output_dir=out_dir,
            filter_name="F555W",
            zeropoint=25.0,
            exptime=500.0,
            aperture_radii=[1.0, 3.0, 4.0],
            user_aperture=4.0,
            photometry_dir=out_dir,
        )

        if not txt_path.exists():
            print("ERROR: expected output missing:", txt_path, file=sys.stderr)
            return 1

        text = txt_path.read_text()
        if not text.strip():
            print("ERROR: output .txt is empty:", txt_path, file=sys.stderr)
            return 1

        mag_path = out_dir / f"mag_{frame.stem}.mag"
        print("OK: photometry finished.")
        print("    .mag:", mag_path if mag_path.exists() else "(missing)")
        print("    .txt:", txt_path)
        print("    first lines of .txt:")
        for line in text.strip().splitlines()[:5]:
            print("   ", line)
        return 0
    except ImportError as e:
        print("ERROR (pyraf/IRAF import):", e, file=sys.stderr)
        print("Install: pip install pyraf; set export IRAF=/path/to/iraf", file=sys.stderr)
        return 2
    except Exception as e:
        print("ERROR:", type(e).__name__, e, file=sys.stderr)
        return 1
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        elif args.keep:
            print("==> Kept outputs under:", work)


if __name__ == "__main__":
    raise SystemExit(main())
