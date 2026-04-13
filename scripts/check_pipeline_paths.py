#!/usr/bin/env python3
"""
Pre-flight check: validate paths and required files before running the pipeline on HPC.

Reads the same env vars as run_pipeline.py (COMP_MAIN_DIR, COMP_FITS_PATH, COMP_PSF_PATH,
COMP_BAO_PATH, COMP_SLUG_LIB_DIR, IRAF). Run this on the login node before qsub to catch
missing paths or files.

Usage:
    # Use env from current shell (e.g. after sourcing PBS script vars)
    python scripts/check_pipeline_paths.py

    # Or with explicit paths (override env)
    python scripts/check_pipeline_paths.py --main-dir /g/data/jh2/xxx --fits-path /g/data/jh2/yyy

    # Include 5-filter photometry checks (FITS per filter, header_info, readme, IRAF)
    python scripts/check_pipeline_paths.py --run-photometry

    # If the galaxy directory is missing HLSP / white / r2 config, run setup_legus_galaxy.py first
    python scripts/check_pipeline_paths.py --galaxy ngc1566 --setup-if-missing

Exit: 0 if all checks pass, 1 if any required path/file is missing.
"""
from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GALAXY_DEFAULT = "ngc628-c"


def _print_setup_legus_failure(returncode: int) -> None:
    """Explain setup_legus_galaxy.py failure (especially SIGKILL / OOM)."""
    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"ERROR: setup_legus_galaxy.py exited with code {returncode}.", flush=True)
    if returncode < 0:
        sig = -returncode
        try:
            sname = signal.Signals(sig).name
        except ValueError:
            sname = f"SIGNAL_{sig}"
        print(f"  Cause: child killed by Unix signal {sig} ({sname}), not a Python exception.", flush=True)
        if sig == 9:
            print(
                "  SIGKILL (9) on HPC usually means:\n"
                "    • OOM killer — white-light step loads several large drizzled FITS into RAM\n"
                "    • Node policy / cgroup memory limit\n"
                "  Fix: run on a PBS job with large --mem (try 64GB–128GB+), or only run white on compute:\n"
                "    source .venv/bin/activate\n"
                "    python scripts/make_white_light.py --galaxy <gal> --project-root <PROJECT_ROOT>\n"
                "  If downloads already finished, you do not need full setup_legus_galaxy.py again.",
                flush=True,
            )
    else:
        print(
            "  Non-zero exit: see traceback above from setup_legus_galaxy.py (if any).",
            flush=True,
        )
    print("=" * 60, flush=True)


def _galaxy_has_hlsp_science_fits(gal_dir: Path) -> bool:
    if not gal_dir.is_dir():
        return False
    for p in gal_dir.rglob("*.fits"):
        if "downloads" in p.parts:
            continue
        n = p.name.lower()
        if "white" in n:
            continue
        if n.startswith("hlsp_legus") or (n.startswith("hlsp") and "legus" in n):
            return True
    return False


def _legus_galaxy_inputs_ready(galaxy_id: str, gal_dir: Path) -> bool:
    """Same readiness criteria as run_pipeline.ensure_legus_galaxy_setup."""
    white = gal_dir / f"{galaxy_id}_white.fits"
    r2 = gal_dir / f"r2_wl_aa_{galaxy_id}.config"
    return (
        _galaxy_has_hlsp_science_fits(gal_dir)
        and white.is_file()
        and r2.is_file()
    )


def _resolve_galaxy_data_dir(galaxy_id: str, fits_path: Path, main_dir: Path) -> Path:
    """Directory that should hold HLSP + white + r2 (create if missing after setup)."""
    gal_short = galaxy_id.split("_")[0]
    for base in (fits_path, main_dir):
        for sub in (galaxy_id, f"{gal_short}_white-R17v100"):
            d = base / sub
            if d.is_dir():
                return d
    return fits_path / galaxy_id


def bootstrap_legus_galaxy_if_needed(
    galaxy_id: str,
    main_dir: Path,
    fits_path: Path,
) -> None:
    """
    Run scripts/setup_legus_galaxy.py when HLSP / white / r2 are missing under the galaxy dir.
    Puts <galaxy>/ under --project-root fits_path (typical: same as main_dir on HPC).
    """
    gal_dir = _resolve_galaxy_data_dir(galaxy_id, fits_path, main_dir)
    if _legus_galaxy_inputs_ready(galaxy_id, gal_dir):
        print(f"[setup-if-missing] Galaxy data OK: {gal_dir}")
        return
    setup_script = main_dir / "scripts" / "setup_legus_galaxy.py"
    if not setup_script.is_file():
        raise FileNotFoundError(f"Cannot bootstrap: missing {setup_script}")
    # Parent of <galaxy>/ must receive downloads (align with COMP_FITS_PATH)
    project_root = fits_path.resolve()
    print(
        f"[setup-if-missing] Running setup_legus_galaxy.py for {galaxy_id!r} "
        f"(project-root={project_root}) ...",
        flush=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(setup_script),
            "--galaxy",
            galaxy_id,
            "--project-root",
            str(project_root),
        ],
        cwd=str(main_dir),
    )
    if proc.returncode != 0:
        _print_setup_legus_failure(proc.returncode)
        raise RuntimeError(
            f"setup_legus_galaxy.py failed (exit {proc.returncode}); diagnosis printed above."
        )
    gal_dir = _resolve_galaxy_data_dir(galaxy_id, fits_path, main_dir)
    if not _legus_galaxy_inputs_ready(galaxy_id, gal_dir):
        raise RuntimeError(
            f"After setup_legus_galaxy.py, expected HLSP + white + r2 under {gal_dir} "
            f"(check COMP_FITS_PATH vs --main-dir)."
        )
    print(f"[setup-if-missing] Done: {gal_dir}", flush=True)


def _path_env(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).resolve() if raw else default.resolve()


def check_path(path: Path, label: str, must_exist: bool = True, must_be_dir: bool = True) -> bool:
    if not path.exists():
        if must_exist:
            print(f"  [MISS] {label}: {path}")
            return False
        print(f"  [  --] {label}: {path} (optional, not found)")
        return True
    if must_be_dir and not path.is_dir():
        print(f"  [MISS] {label}: {path} (not a directory)")
        return False
    print(f"  [ OK ] {label}: {path}")
    return True


def check_file(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"  [MISS] {label}: {path}")
        return False
    if not path.is_file():
        print(f"  [MISS] {label}: {path} (not a file)")
        return False
    print(f"  [ OK ] {label}: {path}")
    return True


def check_any_glob(pattern: str, label: str) -> bool:
    matches = glob.glob(pattern)
    if not matches:
        print(f"  [MISS] {label}: no match for {pattern}")
        return False
    print(f"  [ OK ] {label}: found {len(matches)} match(es) e.g. {matches[0]}")
    return True


def run_checks(
    main_dir: Path,
    fits_path: Path,
    psf_path: Path,
    bao_path: Path,
    slug_lib_dir: Path,
    galaxy_id: str,
    run_photometry: bool,
    iraf_path: str | None,
) -> bool:
    ok = True

    print("--- Project & env paths ---")
    ok &= check_path(main_dir, "main_dir (COMP_MAIN_DIR)", must_exist=True, must_be_dir=True)
    ok &= check_path(fits_path, "fits_path (COMP_FITS_PATH)", must_exist=True, must_be_dir=True)
    ok &= check_path(psf_path, "psf_path (COMP_PSF_PATH)", must_exist=True, must_be_dir=True)
    ok &= check_path(bao_path, "bao_path (COMP_BAO_PATH)", must_exist=True, must_be_dir=True)
    ok &= check_path(slug_lib_dir, "slug_lib_dir (COMP_SLUG_LIB_DIR)", must_exist=True, must_be_dir=True)

    print("\n--- Required files under main_dir ---")
    ok &= check_file(main_dir / "galaxy_names.npy", "galaxy_names.npy")
    ok &= check_file(main_dir / "galaxy_filter_dict.npy", "galaxy_filter_dict.npy")
    ok &= check_file(main_dir / "output.param", "output.param")
    ok &= check_file(main_dir / "default.nnw", "default.nnw")

    # BAOlab executable
    bl = bao_path / "bl"
    bl_exe = bao_path / "bl.exe"
    if not bl.exists() and not bl_exe.exists():
        print(f"  [MISS] BAOlab binary: {bl} or {bl_exe}")
        ok = False
    else:
        print(f"  [ OK ] BAOlab binary: {bl}" if bl.exists() else f"  [ OK ] BAOlab binary: {bl_exe}")

    # At least one PSF file
    psf_glob = str(psf_path / "psf_*.fits")
    if not glob.glob(psf_glob):
        print(f"  [MISS] PSF files: no psf_*.fits in {psf_path}")
        ok = False
    else:
        print(f"  [ OK ] PSF files: psf_*.fits in {psf_path}")

    # SLUG library: expect something under slug_lib_dir
    try:
        next(slug_lib_dir.iterdir())
        print(f"  [ OK ] SLUG library: non-empty {slug_lib_dir}")
    except StopIteration:
        print(f"  [MISS] SLUG library: empty dir {slug_lib_dir}")
        ok = False

    # White / science frame: COMP_SCIFRAME or fits_path/galaxy/ngc628-c_white.fits or similar
    sciframe_env = os.environ.get("COMP_SCIFRAME")
    if sciframe_env:
        ok &= check_file(Path(sciframe_env).resolve(), "white FITS (COMP_SCIFRAME)")
    else:
        gal_short = galaxy_id.split("_")[0]
        # Common layouts: fits_path/galaxy_id or fits_path/galaxy_id_white-R17v100
        for sub in (galaxy_id, f"{gal_short}_white-R17v100", gal_short):
            d = fits_path / sub
            if d.is_dir():
                white_candidates = list(d.glob("*white*.fits")) or list(d.glob("hlsp_legus*.fits"))
                if white_candidates:
                    print(f"  [ OK ] white/science FITS: e.g. {white_candidates[0]}")
                    break
        else:
            print(f"  [MISS] white/science FITS: no *white*.fits or hlsp_legus*.fits under {fits_path}/{galaxy_id} or {fits_path}/{gal_short}_white-R17v100")
            ok = False

    # SExtractor config: in fits_path/galaxy or main_dir/galaxy or fits_path (if fits_path is galaxy dir)
    gal_short = galaxy_id.split("_")[0]
    for candidate in [
        fits_path / galaxy_id / f"r2_wl_aa_{gal_short}.config",
        main_dir / galaxy_id / f"r2_wl_aa_{gal_short}.config",
        fits_path / f"{gal_short}_white-R17v100" / f"r2_wl_aa_{gal_short}.config",
        fits_path / f"r2_wl_aa_{gal_short}.config",
    ]:
        if candidate.exists():
            sex_config = candidate
            break
    else:
        sex_config = main_dir / galaxy_id / f"r2_wl_aa_{gal_short}.config"  # for error message
    ok &= check_file(sex_config, f"SExtractor config r2_wl_aa_{gal_short}.config")

    if not run_photometry:
        print("\n--- (Skip 5-filter checks; use --run-photometry to check) ---")
        if iraf_path:
            check_path(Path(iraf_path), "IRAF (optional)", must_exist=False, must_be_dir=True)
        return ok

    print("\n--- 5-filter photometry inputs ---")
    # Galaxy data dir: same as inject logic (fits_path may point to parent or to galaxy subdir)
    gal_data_candidates = [fits_path / galaxy_id, fits_path / f"{gal_short}_white-R17v100", fits_path]
    gal_data_dir = None
    for d in gal_data_candidates:
        if d.is_dir() and (d / f"header_info_{gal_short}.txt").exists():
            gal_data_dir = d
            break
    if gal_data_dir is None:
        print(f"  [MISS] galaxy data dir: need {fits_path}/{galaxy_id} or .../header_info_{gal_short}.txt")
        ok = False
    else:
        print(f"  [ OK ] galaxy data dir: {gal_data_dir}")

    if gal_data_dir is not None:
        try:
            import numpy as np
            gal_filters = np.load(main_dir / "galaxy_filter_dict.npy", allow_pickle=True).item()
            filters_list, _ = gal_filters.get(gal_short, ([], []))
            filters = list(filters_list)[:5] if filters_list else []
        except Exception as e:
            print(f"  [MISS] galaxy_filter_dict.npy: {e}")
            ok = False
            filters = []
        for filt in filters:
            pattern = str(gal_data_dir / f"hlsp_legus_hst_*{gal_short}_{filt}_*drc.fits")
            if not glob.glob(pattern):
                pattern = str(gal_data_dir / f"*{filt}*.fits")
            ok &= check_any_glob(pattern, f"FITS filter {filt}")
        ok &= check_file(gal_data_dir / f"header_info_{gal_short}.txt", f"header_info_{gal_short}.txt")
        readme_glob = str(gal_data_dir / f"automatic_catalog*_{gal_short}.readme")
        if glob.glob(readme_glob):
            print(f"  [ OK ] readme: automatic_catalog*_{gal_short}.readme")
        else:
            print(f"  [  --] readme: {readme_glob} (optional, use default aperture)")

    if iraf_path:
        ok &= check_path(Path(iraf_path), "IRAF (for photometry)", must_exist=True, must_be_dir=True)
    else:
        print("  [  --] IRAF: not set (required for --run_photometry)")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight check paths and files before running the pipeline on HPC."
    )
    parser.add_argument(
        "--main-dir",
        type=Path,
        default=None,
        help="Project root (default: COMP_MAIN_DIR or script repo root)",
    )
    parser.add_argument(
        "--fits-path",
        type=Path,
        default=None,
        help="FITS root (default: COMP_FITS_PATH or main_dir)",
    )
    parser.add_argument(
        "--psf-path",
        type=Path,
        default=None,
        help="PSF directory (default: COMP_PSF_PATH or main_dir/PSF_files)",
    )
    parser.add_argument(
        "--bao-path",
        type=Path,
        default=None,
        help="BAOlab bin directory (default: COMP_BAO_PATH or main_dir/.deps/local/bin)",
    )
    parser.add_argument(
        "--slug-lib-dir",
        type=Path,
        default=None,
        help="SLUG library (default: COMP_SLUG_LIB_DIR or main_dir/SLUG_library)",
    )
    parser.add_argument(
        "--galaxy",
        type=str,
        default=GALAXY_DEFAULT,
        help=f"Galaxy id (default: {GALAXY_DEFAULT})",
    )
    parser.add_argument(
        "--run-photometry",
        action="store_true",
        help="Also check 5-filter FITS, header_info, readme, IRAF",
    )
    parser.add_argument(
        "--setup-if-missing",
        action="store_true",
        help=(
            "If HLSP science FITS, <galaxy>_white.fits, or r2_wl_aa_<galaxy>.config are missing, "
            "run scripts/setup_legus_galaxy.py (STScI download) before checks. "
            "Uses --project-root = fits_path (usually project root on HPC)."
        ),
    )
    args = parser.parse_args()

    main_dir = args.main_dir or _path_env("COMP_MAIN_DIR", ROOT)
    main_dir = main_dir.resolve()
    fits_path = args.fits_path or _path_env("COMP_FITS_PATH", main_dir)
    fits_path = fits_path.resolve()
    psf_path = args.psf_path or _path_env("COMP_PSF_PATH", main_dir / "PSF_files")
    psf_path = psf_path.resolve()
    bao_path = args.bao_path or _path_env("COMP_BAO_PATH", main_dir / ".deps" / "local" / "bin")
    bao_path = bao_path.resolve()
    slug_lib_dir = args.slug_lib_dir or _path_env("COMP_SLUG_LIB_DIR", main_dir / "SLUG_library")
    slug_lib_dir = slug_lib_dir.resolve()
    iraf_path = os.environ.get("IRAF")

    print("check_pipeline_paths.py — pre-flight path/file check")
    print("=" * 60)
    if args.setup_if_missing:
        bootstrap_legus_galaxy_if_needed(
            galaxy_id=args.galaxy,
            main_dir=main_dir,
            fits_path=fits_path,
        )

    success = run_checks(
        main_dir=main_dir,
        fits_path=fits_path,
        psf_path=psf_path,
        bao_path=bao_path,
        slug_lib_dir=slug_lib_dir,
        galaxy_id=args.galaxy,
        run_photometry=args.run_photometry,
        iraf_path=iraf_path,
    )
    print("=" * 60)
    if success:
        print("All checks passed. You can run the pipeline / qsub.")
        return 0
    print("Some checks failed. Fix paths or env (e.g. in PBS script) and re-run this check.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
