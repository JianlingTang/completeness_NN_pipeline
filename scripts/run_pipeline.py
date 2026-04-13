#!/usr/bin/env python3
"""
Small end-to-end test of stages 1–3 for ngc628-c (white injection).

Phase A: Run generate_white_clusters.py (legacy injection) with small params.
Phase B: Run refactored pipeline stages 2–3 (detection + matching), optionally
  4–5 (photometry + catalogue). By default photometry runs; use --no_photometry to stop at matching. Matched clusters are
  injected onto HLSP 5-filter science images (same coords as white) via
  scripts/inject_clusters_to_5filters.py; photometry and CI cut run on those
  frames, not on white synthetic images.

Usage:
    python scripts/run_pipeline.py                         # SLUG-sampled positions
    python scripts/run_pipeline.py --input_coords FILE     # user-supplied positions
    python scripts/run_pipeline.py --run_ml                # pipeline then build_ml_inputs + perform_ml_to_learn_completeness

If the galaxy directory under the project root is missing HLSP science FITS, white image, or
r2 SExtractor config, run_pipeline automatically runs scripts/setup_legus_galaxy.py (download +
extract + white). Use --skip-galaxy-setup to disable. --check-only does not run setup.
"""
import argparse
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
GALAXY = "ngc628-c"
OUTNAME = "test"
# Sampling: each (frame, reff) gets NCL clusters from SLUG (indices (ridx*nframe+i_frame)*ncl : +ncl).
# So each reff plot aggregates nframe frames → nframe*NCL points (e.g. 3*500=1500).
NCL = 500
NFRAME = 1
ERADIUS = 3
MRMODEL = "flat"
DMOD = 29.98

# Canonical 5-band order (wavelength 275, 336, 435, 555, 814 nm). physprop mag_VEGA_select *.npy
# and SLUG/generate_white_clusters use this column order; plot labels must match.
CANONICAL_5BAND_ORDER = ("F275W", "F336W", "F435W", "F555W", "F814W")


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
                "  SIGKILL (9) on HPC usually means OOM killer or memory cgroup — the white-light step\n"
                "  loads several large drizzled FITS. Use PBS with large --mem or run:\n"
                "    python scripts/make_white_light.py --galaxy <gal> --project-root <ROOT>\n"
                "  on a compute node.",
                flush=True,
            )
    else:
        print("  Non-zero exit: see any traceback above from the child process.", flush=True)
    print("=" * 60, flush=True)


def _galaxy_has_hlsp_science_fits(gal_dir: Path) -> bool:
    """
    True if at least one LEGUS HLSP science FITS exists under gal_dir (recursive),
    excluding downloads/ and *_white*.fits. Matches setup_legus_galaxy.galaxy_has_hlsp_science_fits.
    """
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


def ensure_legus_galaxy_setup(galaxy: str) -> None:
    """
    If HLSP FITS, <galaxy>_white.fits, or r2_wl_aa_<galaxy>.config are missing under ROOT/<galaxy>,
    run scripts/setup_legus_galaxy.py to create the directory, download, extract, sync catalog files,
    and build white (setup script skips steps that are already satisfied).
    """
    gal_dir = ROOT / galaxy
    white_path = gal_dir / f"{galaxy}_white.fits"
    r2_path = gal_dir / f"r2_wl_aa_{galaxy}.config"
    ready = (
        _galaxy_has_hlsp_science_fits(gal_dir)
        and white_path.is_file()
        and r2_path.is_file()
    )
    if ready:
        print(f"[galaxy] LEGUS inputs OK: {gal_dir} (HLSP FITS, {white_path.name}, {r2_path.name})")
        return
    setup_script = ROOT / "scripts" / "setup_legus_galaxy.py"
    if not setup_script.is_file():
        raise FileNotFoundError(f"Cannot bootstrap galaxy: missing {setup_script}")
    print(
        "[galaxy] Missing HLSP FITS, white image, and/or r2 config — running "
        f"{setup_script.relative_to(ROOT)} ...",
        flush=True,
    )
    proc = subprocess.run(
        [PYTHON, str(setup_script), "--galaxy", galaxy, "--project-root", str(ROOT)],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        _print_setup_legus_failure(proc.returncode)
        raise RuntimeError(
            f"setup_legus_galaxy.py failed (exit {proc.returncode}); diagnosis printed above."
        )
    if not white_path.is_file():
        raise FileNotFoundError(f"After setup, expected white FITS not found: {white_path}")
    if not _galaxy_has_hlsp_science_fits(gal_dir):
        raise RuntimeError(f"After setup, no HLSP science FITS found under {gal_dir}")
    if not r2_path.is_file():
        raise FileNotFoundError(f"After setup, expected SExtractor config not found: {r2_path}")
    print(f"[galaxy] Setup finished; using {gal_dir}", flush=True)


def ensure_white_sciframe(galaxy: str, explicit_sciframe: str | None) -> Path:
    """
    Return the white science frame path for Phase A.
    If --sciframe is not provided, (re)generate white via pure Python and use it.
    """
    if explicit_sciframe:
        p = Path(explicit_sciframe).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--sciframe not found: {p}")
        return p

    white_path = ROOT / galaxy / f"{galaxy}_white.fits"
    if white_path.exists():
        print(f"Reusing existing white science frame: {white_path}")
        return white_path
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "make_white_light.py"),
        "--galaxy",
        galaxy,
        "--project-root",
        str(ROOT),
    ]
    print(f"Generating white science frame for {galaxy}...")
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not white_path.exists():
        raise FileNotFoundError(f"White science frame was not created: {white_path}")
    return white_path


def cleanup_all_pipeline_outputs():
    """Remove all pipeline-generated outputs so a fresh run can be done."""
    import shutil
    white_dir = ROOT / GALAXY / "white"
    physprop_dir = ROOT / "physprop"
    tmp_dir = ROOT / "tmp_pipeline_test"

    # physprop
    if physprop_dir.exists():
        for f in physprop_dir.glob("*.npy"):
            f.unlink()
        print("  Removed physprop/*.npy")

    # white
    for name in ["detection_labels", "diagnostics", "matched_coords", "baolab", "synthetic_fits", "synthetic_frames"]:
        d = white_dir / name
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed {d.relative_to(ROOT)}")
    for f in set(white_dir.glob("white_position_*.txt")) | set(white_dir.glob("*_position_*_test_*.txt")):
        f.unlink()
        print(f"  Removed {f.name}")
    for f in white_dir.glob("diagnostic_*.pdf"):
        f.unlink()
        print(f"  Removed {f.name}")
    cat_dir = white_dir / "catalogue"
    if cat_dir.exists():
        for f in cat_dir.glob("*.parquet"):
            f.unlink()
        print("  Removed catalogue/*.parquet")

    # tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"  Removed {tmp_dir.relative_to(ROOT)}")

    # per-filter dirs under GALAXY (e.g. F275w, F336w, ...)
    gal_dir = ROOT / GALAXY
    if gal_dir.exists():
        for sub in gal_dir.iterdir():
            if sub.is_dir() and sub.name != "white":
                for name in ["synthetic_fits", "baolab", "synthetic_frames", "photometry", "s_extraction"]:
                    d = sub / name
                    if d.exists():
                        shutil.rmtree(d)
                        print(f"  Removed {d.relative_to(ROOT)}")
    # white photometry outputs if any (e.g. s_extraction)
    for name in ["s_extraction"]:
        d = white_dir / name
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed {d.relative_to(ROOT)}")
    print("Cleanup done.\n")


def cleanup_heavy_intermediate_dirs_after_artifacts(galaxy: str) -> None:
    """
    Remove heavy intermediate files after key artifacts are saved.
    Keep physprop/*.npy and ML artifacts (det_3d_*.npy / allprop_*.npz).
    """
    import shutil

    def _clear_dir_contents(d: Path) -> int:
        if not d.exists() or not d.is_dir():
            return 0
        removed = 0
        for p in d.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            removed += 1
        return removed

    gal_dir = ROOT / galaxy
    targets = []
    white_dir = gal_dir / "white"
    for name in ("baolab", "synthetic_frames", "synthetic_fits", "photometry", "s_extraction"):
        targets.append(white_dir / name)
    if gal_dir.exists():
        for sub in gal_dir.iterdir():
            if sub.is_dir() and sub.name != "white":
                for name in ("baolab", "synthetic_frames", "synthetic_fits", "photometry", "s_extraction"):
                    targets.append(sub / name)

    total_removed = 0
    touched = 0
    for d in targets:
        n = _clear_dir_contents(d)
        if n > 0:
            touched += 1
            total_removed += n
            print(f"  Cleaned {n} item(s) in {d.relative_to(ROOT)}")
    if touched == 0:
        print("  No heavy intermediate directories needed cleaning.")
    else:
        print(f"Post-run cleanup done: removed {total_removed} item(s) from {touched} directories.\n")


def run_phase_a(
    input_coords: str = None,
    ncl: int = NCL,
    nframe: int = NFRAME,
    reff_list: list = None,
    sigma_pc: float = 100.0,
    sciframe_path: str | None = None,
    placement_mode: str = "white",
    placement_fits: str | None = None,
    exclude_region_param: list[float] | None = None,
):
    """Phase A: inject synthetic clusters via generate_white_clusters.py."""
    import os
    def _p(key: str, default: Path) -> Path:
        raw = os.environ.get(key)
        return Path(raw) if raw else default

    if reff_list is None:
        reff_list = [float(ERADIUS)]
    # Input paths from env (outputs stay under ROOT)
    slug_lib_dir = _p("COMP_SLUG_LIB_DIR", ROOT / "SLUG_library")
    psf_path = _p("COMP_PSF_PATH", ROOT / "PSF_files")
    bao_path = _p("COMP_BAO_PATH", ROOT / ".deps" / "local" / "bin")
    fits_path = _p("COMP_FITS_PATH", ROOT)
    sciframe = Path(sciframe_path).resolve()

    print("=" * 60)
    print("Phase A: Injection (generate_white_clusters.py)")
    print("=" * 60)

    cmd = [
        PYTHON, str(ROOT / "scripts" / "generate_white_clusters.py"),
        "--ncl", str(ncl),
        "--nframe", str(nframe),
        "--eradius_list",
    ] + [str(int(r)) for r in reff_list] + [
        "--gal_name", GALAXY,
        "--galaxy_fullname", GALAXY,
        "--sciframe", str(sciframe),
        "--directory", str(ROOT),
        "--mrmodel", MRMODEL,
        "--dmod", str(DMOD),
        "--outname", OUTNAME,
        "--fits_path", str(fits_path),
        "--psf_path", str(psf_path),
        "--bao_path", str(bao_path),
        "--slug_lib_dir", str(slug_lib_dir),
        "--sigma_pc", str(sigma_pc),
        "--placement_mode", placement_mode,
    ]
    if placement_fits is not None:
        cmd.extend(["--placement_fits", str(Path(placement_fits).resolve())])
    if exclude_region_param is not None:
        cmd.append("--exclude_region_param")
        cmd.extend(str(x) for x in exclude_region_param)
    if input_coords is not None:
        cmd.extend(["--input_coords", str(Path(input_coords).resolve())])
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"Phase A FAILED (exit {result.returncode})")
        sys.exit(1)
    print("Phase A completed successfully.\n")


def verify_phase_a_outputs(ncl: int = NCL, nframe: int = NFRAME, reff_list: list = None):
    """Check that Phase A produced expected files for all frames and reff."""
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    white_dir = ROOT / GALAXY / "white"
    synth_dir = white_dir / "synthetic_fits"
    missing = []
    for i_frame in range(nframe):
        for reff_f in reff_list:
            r = float(reff_f)
            coord_file = white_dir / f"white_position_{i_frame}_{OUTNAME}_reff{r:.2f}.txt"
            if not coord_file.exists():
                missing.append(str(coord_file))
    expected = nframe * len(reff_list)
    synth_frames = []
    for reff_f in reff_list:
        r = float(reff_f)
        synth_frames.extend(synth_dir.glob(f"*_frame*_{OUTNAME}_reff{r:.2f}.fits"))
    print("Verifying Phase A outputs:")
    print(f"  Coord files (frames 0..{nframe - 1} × {len(reff_list)} reff): {expected - len(missing)}/{expected} found")
    print(f"  Synth frames: {len(synth_frames)} found")
    if missing:
        print("ERROR: missing coord files:", missing[:3], "..." if len(missing) > 3 else "")
        sys.exit(1)
    if len(synth_frames) < expected:
        print("ERROR: expected at least", expected, "synthetic FITS frames")
        sys.exit(1)
    print("Phase A outputs verified.\n")
    return None, synth_frames


def prepare_five_filter_frames_for_photometry(nframe: int = NFRAME):
    """
    Copy white synthetic frames into each filter's synthetic_fits so that
    five-filter photometry can run for all frames.
    """
    import shutil

    import numpy as np
    white_dir = ROOT / GALAXY / "white"
    synth_dir = white_dir / "synthetic_fits"
    reff_f = float(ERADIUS)
    gal_filters = np.load(ROOT / "galaxy_filter_dict.npy", allow_pickle=True).item()
    gal_short = GALAXY.split("_")[0]
    filters_list, _ = gal_filters.get(gal_short, ([], []))
    filters = sorted(filters_list) if filters_list else []
    for i_frame in range(nframe):
        frame_pattern = f"*_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.fits"
        white_frames = list(synth_dir.glob(frame_pattern))
        if not white_frames:
            continue
        src = white_frames[0]
        for filt in filters:
            dest_dir = ROOT / GALAXY / filt / "synthetic_fits"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"{GALAXY}_{filt}_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.fits"
            dest = dest_dir / dest_name
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
                print(f"  Prepared {dest.relative_to(ROOT)}")
    print("Five-filter synthetic_fits prepared.\n")


def run_phase_b(ncl: int = NCL, nframe: int = NFRAME, run_photometry: bool = False, reff_list: list = None, parallel: bool = False, n_workers: int | None = None, delete_synthetic_after_use: bool = False):
    """Phase B: run refactored pipeline stages 2–3 (detection + matching), optionally 4–5 (photometry + catalogue)."""
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    reff_list = [float(r) for r in reff_list]
    # Allow parallel + photometry for this test run
    use_parallel = parallel
    if run_photometry and parallel:
        print("Note: Testing parallel + photometry.", flush=True)
    print("=" * 60)
    print("Phase B: Detection + Matching" + (" + Photometry + Catalogue" if run_photometry else ""))
    print("=" * 60)
    n_jobs = nframe * len(reff_list)
    print(f"Will process {n_jobs} job(s) (frame 0..{nframe - 1} × reff {reff_list}).")
    if use_parallel and n_jobs > 1:
        import multiprocessing as mp
        w = n_workers if n_workers is not None else min(n_jobs, max(1, mp.cpu_count() - 1))
        print(f"Parallel: {w} workers.", flush=True)
    if run_photometry:
        print("Photometry: 5 filters per frame, each ~1–2 min — total can be 10+ min per frame.", flush=True)
    print("", flush=True)

    from cluster_pipeline.config import PipelineConfig
    from cluster_pipeline.pipeline import run_galaxy_pipeline

    import os
    def _path_env(key: str, default: Path) -> Path:
        raw = os.environ.get(key)
        return Path(raw) if raw else default

    # Outputs under project root (ROOT). Inputs from env.
    fits_path = _path_env("COMP_FITS_PATH", ROOT)
    sex_config = (fits_path / GALAXY / f"r2_wl_aa_{GALAXY}.config")
    if not sex_config.exists():
        sex_config = ROOT / GALAXY / f"r2_wl_aa_{GALAXY}.config"
    param_file = ROOT / "output.param"
    nnw_file = ROOT / "default.nnw"

    cfg_kw = dict(
        main_dir=ROOT,
        fits_path=fits_path,
        psf_path=_path_env("COMP_PSF_PATH", ROOT / "PSF_files"),
        bao_path=_path_env("COMP_BAO_PATH", ROOT / ".deps" / "local" / "bin"),
        slug_lib_dir=_path_env("COMP_SLUG_LIB_DIR", ROOT / "SLUG_library"),
        output_lib_dir=_path_env("COMP_OUTPUT_LIB_DIR", ROOT / "SLUG_library"),
        temp_base_dir=_path_env("COMP_TEMP_BASE_DIR", ROOT / "tmp_pipeline_test"),
        ncl=ncl,
        nframe=nframe,
        reff_list=reff_list,
        mrmodel=MRMODEL,
        dmod=DMOD,
        sextractor_config_path=sex_config,
        sextractor_param_path=param_file,
        sextractor_nnw_path=nnw_file,
    )
    if run_photometry:
        inject_script = ROOT / "scripts" / "inject_clusters_to_5filters.py"
        if inject_script.exists():
            cfg_kw["inject_5filter_script"] = inject_script
        else:
            print("WARNING: inject_clusters_to_5filters.py not found; photometry will expect pre-existing synthetic_fits per filter.", flush=True)
    cfg = PipelineConfig(**cfg_kw)

    max_stage = 5 if run_photometry else 3
    run_galaxy_pipeline(
        galaxy_id=GALAXY,
        config=cfg,
        outname=OUTNAME,
        max_stage=max_stage,
        run_photometry=run_photometry,
        run_catalogue=run_photometry,
        keep_frames=not delete_synthetic_after_use,
        delete_synthetic_after_use=delete_synthetic_after_use,
        parallel=use_parallel,
        n_workers=n_workers,
    )
    print("Phase B completed successfully.\n")


def summarise_outputs(ncl: int = NCL, reff_list: list | None = None):
    """Print final output files."""
    print("=" * 60)
    print("Final outputs")
    print("=" * 60)

    white_dir = ROOT / GALAXY / "white"
    dirs_to_check = {
        "Synthetic FITS": white_dir / "synthetic_fits",
        "Matched coords": white_dir / "matched_coords",
        "Diagnostics": white_dir / "diagnostics",
        "Binary labels": white_dir / "detection_labels",
        "Catalogue (photometry + in_catalogue)": white_dir / "catalogue",
    }
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    reff_f2 = float(reff_list[0])
    coord_file = white_dir / f"white_position_0_{OUTNAME}_reff{reff_f2:.2f}.txt"

    print(f"\nCoord file: {coord_file}")
    if coord_file.exists():
        with open(coord_file) as f:
            lines = f.readlines()
        print(f"  {len(lines)} injected clusters")
        if lines:
            print(f"  First line: {lines[0].strip()}")

    for label, d in dirs_to_check.items():
        print(f"\n{label}: {d}")
        if d.exists():
            for p in sorted(d.iterdir()):
                size_kb = p.stat().st_size / 1024
                print(f"  {p.name}  ({size_kb:.1f} KB)")
        else:
            print("  (directory not found)")

    physprop_dir = ROOT / "physprop"
    if physprop_dir.exists():
        npy_files = sorted(physprop_dir.glob("*.npy"))
        if npy_files:
            print(f"\nPhysical properties ({physprop_dir}):")
            for p in npy_files:
                print(f"  {p.name}")


def backfill_physprop_from_white_coords(nframe: int, input_coords_path: str = None, reff_list: list = None):
    """
    Write physprop .npy files for frames that have white_position files but no physprop.
    If reff_list is given, backfill for each (frame, reff); else use single ERADIUS.
    """
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    reff_list = [float(r) for r in reff_list]
    import numpy as np
    white_dir = ROOT / GALAXY / "white"
    physprop_dir = ROOT / "physprop"
    physprop_dir.mkdir(parents=True, exist_ok=True)
    mrmodel = "flat"

    # Prefer 5-column input_coords for 1:1 mass/age with pipeline clusters
    use_file_phys = False
    ic_mass, ic_age = None, None
    if input_coords_path:
        path = Path(input_coords_path).resolve()
        if path.exists():
            _ic = np.loadtxt(path)
            if _ic.ndim == 1:
                _ic = _ic.reshape(1, -1)
            if _ic.shape[1] >= 5:
                ic_mass = np.asarray(_ic[:, 3], dtype=float)
                ic_age = np.maximum(np.asarray(_ic[:, 4], dtype=float), 1e6)
                use_file_phys = True
                print(f"[backfill] Using mass/age from 5-col input_coords (1:1 with pipeline): {path}")
    if not use_file_phys:
        default_ic = ROOT / GALAXY / "white" / "input_coords_500.txt"
        if default_ic.exists():
            _ic = np.loadtxt(default_ic)
            if _ic.ndim == 1:
                _ic = _ic.reshape(1, -1)
            if _ic.shape[1] >= 5:
                ic_mass = np.asarray(_ic[:, 3], dtype=float)
                ic_age = np.maximum(np.asarray(_ic[:, 4], dtype=float), 1e6)
                use_file_phys = True
                print(f"[backfill] Using mass/age from 5-col default input_coords (1:1): {default_ic}")
    if not use_file_phys:
        print("[backfill] No 5-col input_coords; using SLUG cycle (mass/age NOT 1:1 with pipeline).")
        gal_filters = np.load(ROOT / "galaxy_filter_dict.npy", allow_pickle=True).item()
        filters = gal_filters[GALAXY]
        allfilters_cam = []
        for filt, cam in zip(filters[0], filters[1]):
            f, c = filt.upper(), cam.upper()
            if c == "WFC3":
                c = "WFC3_UVIS"
            allfilters_cam.append(f"{c}_{f}")
        allfilters_cam = sorted(allfilters_cam, key=lambda x: x[-4:])
        libdir = ROOT / "SLUG_library"
        libname = str(libdir / "flat_in_logm")
        try:
            from slugpy import read_cluster
        except ImportError:
            from cluster_pipeline.data.slug_reader import read_cluster
        lib = read_cluster(libname, read_filters=allfilters_cam, photsystem="L_lambda")
        mass_pool = np.asarray(lib.actual_mass, dtype=float)
        age_pool = np.asarray(lib.time, dtype=float) - np.asarray(lib.form_time, dtype=float)
        age_pool = np.maximum(age_pool, 1e6)
        pool_size = len(mass_pool)
        pool_offset = [0]

        def take_mass_age(ncl):
            idx = (np.arange(ncl) + pool_offset[0]) % pool_size
            pool_offset[0] = (pool_offset[0] + ncl) % pool_size
            return mass_pool[idx].copy(), age_pool[idx].copy()

    written = 0
    for reff_f in reff_list:
        for i_frame in range(nframe):
            coord_path = white_dir / f"white_position_{i_frame}_{OUTNAME}_reff{reff_f:.2f}.txt"
            if not coord_path.exists():
                continue
            data = np.loadtxt(coord_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            ncl = len(data)
            mag_bao_select = data[:, -1]
            if use_file_phys:
                mass_select = ic_mass[:ncl].copy()
                age_select = ic_age[:ncl].copy()
            else:
                mass_select, age_select = take_mass_age(ncl)
            av_select = np.zeros(ncl)
            mag_vega_select = np.broadcast_to(mag_bao_select.reshape(-1, 1), (ncl, 5))
            base = f"reff{int(reff_f)}_{OUTNAME}"
            to_save = {
                f"mass_select_model{mrmodel}_frame{i_frame}_{base}.npy": mass_select,
                f"age_select_model{mrmodel}_frame{i_frame}_{base}.npy": age_select,
                f"av_select_model{mrmodel}_frame{i_frame}_{base}.npy": av_select,
                f"mag_BAO_select_model{mrmodel}_frame{i_frame}_{base}.npy": mag_bao_select,
                f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_{base}.npy": mag_vega_select,
            }
            for fname, arr in to_save.items():
                np.save(physprop_dir / fname, arr)
            written += 1
    if written:
        print(f"Backfilled physprop for {written} (frame, reff) pairs (mag from white_position).")


def print_catalogue_criterion_counts(nframe: int = 1, reff_list: list = None):
    """Print how many matched clusters each catalogue criterion filters out (for each frame, reff)."""
    import pandas as pd
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    reff_list = [float(r) for r in reff_list]
    white_dir = ROOT / GALAXY / "white"
    cat_dir = white_dir / "catalogue"
    if not cat_dir.exists():
        return
    criteria = ["passes_ci", "passes_stage1_merr", "passes_stage2_merr", "passes_MV"]
    print("\n" + "=" * 60)
    print("Catalogue criteria: clusters filtered out (matched → in_catalogue)")
    print("=" * 60)
    for i_frame in range(nframe):
        for reff_f in reff_list:
            path = cat_dir / f"catalogue_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            n = len(df)
            print(f"\nFrame {i_frame}, reff={reff_f:.2f}: {n} matched clusters")
            for c in criteria:
                if c not in df.columns:
                    continue
                fail = (df[c] == 0).sum()
                pass_n = (df[c] == 1).sum()
                print(f"  {c}: 筛掉 {fail}, 通过 {pass_n}")
            rec = (df["in_catalogue"] == 1).sum()
            print(f"  in_catalogue (recovered): {rec}")
    print()


def plot_recovered_on_white_light(
    galaxy: str,
    outname: str,
    reff_list: list,
    nframe: int,
    main_dir: Path | None = None,
    bottom_zoom_scale: float = 0.6,
    bottom_region_pix: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Reuse scripts/plot_three_panel_white_synthetic_recovered.py to produce
    per-frame recovered overlays as PDF with a modestly zoomed-in bottom panel.
    """
    main_dir = Path(main_dir or ROOT).resolve()
    reff_list = [float(r) for r in reff_list]
    three_panel_script = ROOT / "scripts" / "plot_three_panel_white_synthetic_recovered.py"
    if not three_panel_script.is_file():
        print(
            f"WARNING: skip three-panel recovered plots (missing script): {three_panel_script}",
            flush=True,
        )
        return
    cat_dir = main_dir / galaxy / "white" / "catalogue"
    diag_dir = main_dir / galaxy / "white" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    # If user provides a pixel region, use it directly: (cx, cy, width, height).
    # Otherwise use the script defaults scaled down for a tighter zoom.
    if bottom_region_pix is not None:
        bot_cx, bot_cy, bot_width, bot_height = bottom_region_pix
    else:
        bot_cx = 1763.9872
        bot_cy = 2030.9141
        bot_width = 1117.9524 * float(bottom_zoom_scale)
        bot_height = 587.64165 * float(bottom_zoom_scale)

    for i_frame in range(nframe):
        for reff_f in reff_list:
            mp = cat_dir / f"match_results_frame{i_frame}_{outname}_reff{reff_f:.2f}.parquet"
            cp = cat_dir / f"catalogue_frame{i_frame}_{outname}_reff{reff_f:.2f}.parquet"
            if not mp.is_file() or not cp.is_file():
                print(
                    f"WARNING: skip recovered-on-white frame={i_frame} reff={reff_f:.2f} "
                    f"(missing {mp.name} or {cp.name})",
                    flush=True,
                )
                continue
            out_pdf = diag_dir / (
                f"three_panel_catalogue_recovered_frame{i_frame}_{outname}_reff{reff_f:.2f}.pdf"
            )
            cmd = [
                PYTHON,
                str(three_panel_script),
                "--galaxy",
                str(galaxy),
                "--outname",
                str(outname),
                "--reff",
                str(reff_f),
                "--frame",
                str(i_frame),
                "--bot-cx",
                str(bot_cx),
                "--bot-cy",
                str(bot_cy),
                "--bot-width",
                str(bot_width),
                "--bot-height",
                str(bot_height),
                "--save",
                str(out_pdf),
            ]
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                print(
                    f"WARNING: three-panel plot failed for frame={i_frame} reff={reff_f:.2f}",
                    flush=True,
                )
                continue
            print(
                f"Saved three-panel recovered PDF to {out_pdf}",
                flush=True,
            )


def plot_completeness_diagnostics(nframe: int = 1, reff_list: list = None):
    """
    Plot completeness (y) vs mass, age, white mag, and mag in 5 bands (x).

    Completeness = (number recovered in bin) / (number injected in bin) for:
      - Main 3x3 panel figure (catalogue recovery or white-match)
      - Figure-6 style recovered_completeness_*.png
      - White-detection-only figure (white-match / injected)
    Exception: "matched only" figure uses denominator = matched in bin (to show catalogue/M_V turnover).

    If reff_list has multiple values, one figure per reff is saved.
    """
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    reff_list = [float(r) for r in reff_list]

    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    white_dir = ROOT / GALAXY / "white"
    diag_dir = white_dir / "diagnostics"
    labels_dir = white_dir / "detection_labels"
    physprop_dir = ROOT / "physprop"
    mrmodel = "flat"
    # Plot 5-band panels in canonical order (275, 336, 435, 555, 814); mag_5 columns must match this order
    filter_names = list(CANONICAL_5BAND_ORDER)

    def binned_completeness(x, y_det, n_bins=30):
        """Completeness per bin = (number recovered in bin) / (number injected in bin).
        x, y_det aligned: one entry per injected cluster; y_det = 0/1 (recovered or not).
        """
        ok = np.isfinite(x)
        x_ok, y_ok = x[ok], y_det[ok]
        if len(x_ok) < 2:
            return np.array([]), np.array([]), np.array([])
        bins = np.percentile(x_ok, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.array([np.median(x_ok)]), np.array([np.mean(y_ok)]), np.array([len(x_ok)])
        hist_total, _ = np.histogram(x_ok, bins=bins)  # injected in bin
        hist_det, _ = np.histogram(x_ok, bins=bins, weights=y_ok)  # recovered in bin
        centers = (bins[:-1] + bins[1:]) / 2
        comp = np.full_like(hist_total, np.nan, dtype=float)
        np.divide(hist_det, hist_total, out=comp, where=hist_total > 0)
        return centers, comp, hist_total

    def binned_completeness_mag(mag, y_det, n_bins=30):
        """Completeness per mag bin = (recovered in bin) / (injected in bin); same alignment as binned_completeness."""
        ok = np.isfinite(mag)
        mag_ok, y_ok = mag[ok], y_det[ok]
        if len(mag_ok) < 2:
            return np.array([]), np.array([]), np.array([])
        lo, hi = np.nanmin(mag_ok), np.nanmax(mag_ok)
        if lo >= hi:
            return np.array([lo]), np.array([np.mean(y_ok)]), np.array([len(mag_ok)])
        bins = np.linspace(lo, hi, n_bins + 1)
        hist_total, _ = np.histogram(mag_ok, bins=bins)
        hist_det, _ = np.histogram(mag_ok, bins=bins, weights=y_ok)
        centers = (bins[:-1] + bins[1:]) / 2
        comp = np.full_like(hist_total, np.nan, dtype=float)
        np.divide(hist_det, hist_total, out=comp, where=hist_total > 0)
        return centers, comp, hist_total

    def binned_completeness_mag_range(mag, y_det, mag_lo=20.0, mag_hi=26.0, n_bins=30):
        """Completeness per mag bin over fixed range; denominator = injected in bin (or matched if y_det is subset)."""
        ok = np.isfinite(mag) & (mag >= mag_lo) & (mag <= mag_hi)
        mag_ok, y_ok = mag[ok], y_det[ok]
        if len(mag_ok) < 2:
            return np.array([]), np.array([])
        bins = np.linspace(mag_lo, mag_hi, n_bins + 1)
        hist_total, _ = np.histogram(mag_ok, bins=bins)
        hist_det, _ = np.histogram(mag_ok, bins=bins, weights=y_ok)
        centers = (bins[:-1] + bins[1:]) / 2
        comp = np.full_like(hist_total, np.nan, dtype=float)
        np.divide(hist_det, hist_total, out=comp, where=hist_total > 0)
        return centers, comp

    for reff_f in reff_list:
        all_labels = []
        all_matched = []  # 1 = matched (got to photometry), 0 = unmatched
        all_mass = []
        all_age = []
        all_mag_white = []
        all_mag5 = []
        n_use_per_frame = []  # for white-detection-only plot alignment
        label_source = ""

        for i_frame in range(nframe):
            final_labels_path = labels_dir / f"detection_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.npy"
            white_match_path = labels_dir / f"detection_labels_white_match_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.npy"
            # Prefer catalogue recovery (in_catalogue) so completeness = recovery rate; fall back to white-match if no catalogue run
            if final_labels_path.exists():
                lab = np.load(final_labels_path)
                if i_frame == 0:
                    label_source = "catalogue recovery (in_catalogue)"
                matched = np.load(white_match_path) if white_match_path.exists() else (lab.copy() if hasattr(lab, 'copy') else np.array(lab))
            elif white_match_path.exists():
                lab = np.load(white_match_path)
                if i_frame == 0:
                    label_source = "detection (white-match)"
                matched = lab
            else:
                continue
            mass_path = physprop_dir / f"mass_select_model{mrmodel}_frame{i_frame}_reff{int(reff_f)}_{OUTNAME}.npy"
            age_path = physprop_dir / f"age_select_model{mrmodel}_frame{i_frame}_reff{int(reff_f)}_{OUTNAME}.npy"
            mag_path = physprop_dir / f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_reff{int(reff_f)}_{OUTNAME}.npy"
            mag_white_path = physprop_dir / f"mag_BAO_select_model{mrmodel}_frame{i_frame}_reff{int(reff_f)}_{OUTNAME}.npy"
            if not mass_path.exists() or not age_path.exists() or not mag_path.exists():
                continue
            mass = np.load(mass_path)
            age = np.load(age_path)
            mag_5 = np.load(mag_path)
            if mag_white_path.exists():
                mag_white = np.load(mag_white_path).ravel()
            else:
                mag_white = np.full(len(mass), np.nan)
            n_use = min(len(lab), len(mass), len(age), len(mag_5), len(mag_white), len(matched))
            if n_use < 5:
                continue
            white_coord_path = white_dir / f"white_position_{i_frame}_{OUTNAME}_reff{reff_f:.2f}.txt"
            if white_coord_path.exists() and n_use > 0:
                wp = np.loadtxt(white_coord_path)
                if wp.ndim == 1:
                    wp = wp.reshape(1, -1)
                if wp.shape[0] >= n_use and wp.shape[1] >= 3:
                    wp_mag = wp[:n_use, 2].astype(float)
                    if np.any(np.isfinite(mag_white[:n_use])) and np.any(np.isfinite(wp_mag)):
                        diff = np.abs(mag_white[:n_use] - wp_mag)
                        if np.nanmax(diff) > 1e-5:
                            print(f"WARNING: white_position mag vs physprop mag_BAO differ (frame {i_frame}, reff={reff_f}) max|diff|={np.nanmax(diff):.6f}")
            all_labels.append(lab[:n_use])
            all_matched.append((matched[:n_use] == 1).astype(np.intp))
            all_mass.append(mass[:n_use])
            all_age.append(age[:n_use])
            all_mag_white.append(mag_white[:n_use])
            all_mag5.append(mag_5[:n_use] if mag_5.ndim >= 2 else mag_5[:n_use].reshape(-1, 1))
            n_use_per_frame.append(n_use)

        if not all_labels:
            print(f"WARNING: No detection labels/physprop found for reff={reff_f}; skipping completeness plot")
            continue
        # Aligned by row index: labels[i], mass[i], mag_5[i] refer to the same cluster (cluster_id order per frame)
        labels = np.concatenate(all_labels)
        matched_mask = np.concatenate(all_matched).astype(bool)
        mass = np.concatenate(all_mass)
        age = np.concatenate(all_age)
        mag_white = np.concatenate(all_mag_white)
        mag_5 = np.concatenate(all_mag5, axis=0)
        if mag_5.ndim == 1:
            mag_5 = mag_5.reshape(-1, 1)
        n = len(labels)
        n_matched = int(np.sum(matched_mask))
        if n < 10:
            print(f"WARNING: Too few clusters for completeness plot (reff={reff_f}, n={n})")
            continue

        # mag_5 columns 0..4 = F275W, F336W, F435W, F555W, F814W (canonical order from physprop/SLUG)
        n_filters = min(5, mag_5.shape[1])
        n_cols = 3
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = axes.flatten()

        cx, cy, ct = binned_completeness(np.log10(np.maximum(mass, 1e-10)), labels, n_bins=30)
        ax = axes[0]
        ax.plot(cx, cy * 100.0, "o-", color="C0")
        ax.set_xlabel(r"$\log_{10}$(mass / M$_\odot$)")
        ax.set_ylabel("Completeness [%]")
        ax.set_title("vs mass")
        ax.set_ylim(-5, 105)
        ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)

        cx, cy, ct = binned_completeness(np.log10(np.maximum(age, 1e-7)), labels, n_bins=30)
        ax = axes[1]
        ax.plot(cx, cy * 100.0, "o-", color="C1")
        ax.set_xlabel(r"$\log_{10}$(age / yr)")
        ax.set_ylabel("Completeness [%]")
        ax.set_title("vs age")
        ax.set_ylim(-5, 105)
        ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        cx, cy, ct = binned_completeness_mag(mag_white, labels, n_bins=30)
        if len(cx) > 0:
            order = np.argsort(cx)
            cx, cy = cx[order], cy[order] * 100.0
        ax.plot(cx, cy, "o-", color="C2")
        ax.set_xlabel("mag (white)")
        ax.set_ylabel("Completeness [%]")
        ax.set_title("vs white mag")
        ax.set_ylim(-5, 105)
        ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)

        for i in range(n_filters):
            ax = axes[3 + i]
            m = mag_5[:, i]
            cx, cy, ct = binned_completeness_mag(m, labels, n_bins=30)
            if len(cx) > 0:
                order = np.argsort(cx)
                cx, cy = cx[order], cy[order] * 100.0
            ax.plot(cx, cy, "o-", color=f"C{(3 + i) % 10}")
            ax.set_xlabel(f"mag ({filter_names[i]})")
            ax.set_ylabel("Completeness [%]")
            ax.set_title(f"vs {filter_names[i]}")
            ax.set_ylim(-5, 105)
            ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
            ax.grid(True, alpha=0.3)
        for j in range(3 + n_filters, len(axes)):
            axes[j].set_visible(False)

        title = f"Completeness ({GALAXY}, reff={reff_f}, n={n} clusters"
        if nframe > 1:
            title += f", {nframe} frames"
        title += f", {label_source})"
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"completeness_all{nframe}frames_{n}cl_{OUTNAME}_reff{reff_f:.2f}.png" if nframe > 1 else f"completeness_frame0_{OUTNAME}_reff{reff_f:.2f}.png"
        out_path = diag_dir / out_name
        fig.savefig(out_path, dpi=120)
        plt.close()
        print(f"Saved completeness diagnostics to {out_path}")

        # Same figure but on white detection only (use white-match labels instead of catalogue recovery)
        if nframe > 1 and len(n_use_per_frame) == nframe and all((labels_dir / f"detection_labels_white_match_frame{i}_{OUTNAME}_reff{reff_f:.2f}.npy").exists() for i in range(nframe)):
            all_labels_wd = []
            for i_frame in range(nframe):
                white_match_path = labels_dir / f"detection_labels_white_match_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.npy"
                lab_wd = np.load(white_match_path)
                n_use_wd = n_use_per_frame[i_frame]
                all_labels_wd.append(lab_wd[:n_use_wd])
            if len(all_labels_wd) == nframe:
                labels_wd = np.concatenate(all_labels_wd)
                n_wd = len(labels_wd)
                fig_wd, axes_wd = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
                axes_wd = axes_wd.flatten()
                cx, cy, ct = binned_completeness(np.log10(np.maximum(mass, 1e-10)), labels_wd, n_bins=30)
                ax = axes_wd[0]
                ax.plot(cx, cy * 100.0, "o-", color="C0")
                ax.set_xlabel(r"$\log_{10}$(mass / M$_\odot$)")
                ax.set_ylabel("Completeness [%]")
                ax.set_title("vs mass")
                ax.set_ylim(-5, 105)
                ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
                ax.grid(True, alpha=0.3)
                cx, cy, ct = binned_completeness(np.log10(np.maximum(age, 1e-7)), labels_wd, n_bins=30)
                ax = axes_wd[1]
                ax.plot(cx, cy * 100.0, "o-", color="C1")
                ax.set_xlabel(r"$\log_{10}$(age / yr)")
                ax.set_ylabel("Completeness [%]")
                ax.set_title("vs age")
                ax.set_ylim(-5, 105)
                ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
                ax.grid(True, alpha=0.3)
                ax = axes_wd[2]
                cx, cy, ct = binned_completeness_mag(mag_white, labels_wd, n_bins=30)
                if len(cx) > 0:
                    order = np.argsort(cx)
                    cx, cy = cx[order], cy[order] * 100.0
                ax.plot(cx, cy, "o-", color="C2")
                ax.set_xlabel("mag (white)")
                ax.set_ylabel("Completeness [%]")
                ax.set_title("vs white mag")
                ax.set_ylim(-5, 105)
                ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
                ax.grid(True, alpha=0.3)
                for i in range(n_filters):
                    ax = axes_wd[3 + i]
                    m = mag_5[:, i]
                    cx, cy, ct = binned_completeness_mag(m, labels_wd, n_bins=30)
                    if len(cx) > 0:
                        order = np.argsort(cx)
                        cx, cy = cx[order], cy[order] * 100.0
                    ax.plot(cx, cy, "o-", color=f"C{(3 + i) % 10}")
                    ax.set_xlabel(f"mag ({filter_names[i]})")
                    ax.set_ylabel("Completeness [%]")
                    ax.set_title(f"vs {filter_names[i]}")
                    ax.set_ylim(-5, 105)
                    ax.axhline(90, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
                    ax.axhline(50, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
                    ax.grid(True, alpha=0.3)
                for j in range(3 + n_filters, len(axes_wd)):
                    axes_wd[j].set_visible(False)
                fig_wd.suptitle(f"Completeness ({GALAXY}, reff={reff_f}, n={n_wd} clusters, {nframe} frames, white detection only)", fontsize=11)
                fig_wd.tight_layout()
                out_name_wd = f"completeness_all{nframe}frames_{n_wd}cl_{OUTNAME}_reff{reff_f:.2f}_white_detection_only.png"
                out_path_wd = diag_dir / out_name_wd
                fig_wd.savefig(out_path_wd, dpi=120)
                plt.close(fig_wd)
                print(f"Saved completeness (white detection only) to {out_path_wd}")

        # LEGUS-original style: two panels "Total recovery" (before CI) and "Total recovery after CIcut" (catalogue)
        # Same logic as legus_original_pipeline.py: bin on simulated (apparent) mag, completeness = recovered/injected.
        # Use V-band apparent mag for x-axis (mag_VEGA_select col 3 = F555W) to match LEGUS recovery on V.
        minmag_legus, maxmag_legus, binsize_legus = 20.0, 26.0, 0.3
        mag_v = mag_5[:, 3] if mag_5.shape[1] > 3 else mag_white.astype(float)  # V band apparent
        mag_plot = np.asarray(mag_v, dtype=float)
        ok_m = np.isfinite(mag_plot) & (mag_plot >= minmag_legus) & (mag_plot <= maxmag_legus)
        mag_ok = mag_plot[ok_m]
        n_inj = len(mag_ok)
        if n_inj >= 5:
            labels_white = matched_mask.astype(np.intp)  # before CI = white detection
            labels_cat = labels  # after CI = catalogue (CI + merr>=4 + M_V)
            w_ok = labels_white[ok_m]
            c_ok = labels_cat[ok_m]
            bins_legus = np.arange(minmag_legus, maxmag_legus + binsize_legus, binsize_legus)
            hist_inj, _ = np.histogram(mag_ok, bins=bins_legus)
            hist_before, _ = np.histogram(mag_ok, bins=bins_legus, weights=w_ok)
            hist_after, _ = np.histogram(mag_ok, bins=bins_legus, weights=c_ok)
            centers_legus = (bins_legus[:-1] + bins_legus[1:]) / 2
            comp_before = np.full_like(hist_inj, np.nan, dtype=float)
            comp_after = np.full_like(hist_inj, np.nan, dtype=float)
            np.divide(hist_before, hist_inj, out=comp_before, where=hist_inj > 0)
            np.divide(hist_after, hist_inj, out=comp_after, where=hist_inj > 0)
            comp_before_pct = comp_before * 100.0
            comp_after_pct = comp_after * 100.0

            def completeness_limit(mag_centers, comp_pct, target):
                for a in range(len(comp_pct) - 1):
                    if comp_pct[a] >= target and comp_pct[a + 1] <= target:
                        k = (comp_pct[a + 1] - comp_pct[a]) / (mag_centers[a + 1] - mag_centers[a])
                        m = comp_pct[a] - k * mag_centers[a]
                        return (target - m) / k if k != 0 else float("nan")
                return float("nan")

            lim90 = completeness_limit(centers_legus, comp_before_pct, 90)
            lim50 = completeness_limit(centers_legus, comp_before_pct, 50)
            lim90_CI = completeness_limit(centers_legus, comp_after_pct, 90)
            lim50_CI = completeness_limit(centers_legus, comp_after_pct, 50)

            fig_legus, ax_legus = plt.subplots(1, 2, figsize=(9, 5))
            # Left: Total recovery (before CI cut)
            ax_legus[0].plot(centers_legus, comp_before_pct, color="grey", linewidth=2)
            ax_legus[0].plot(centers_legus, comp_before_pct, "o", color="grey", markeredgecolor="#4D4D4D")
            ax_legus[0].set_title("Total recovery (before CI cut)")
            ax_legus[0].set_ylabel("Completeness [%]")
            ax_legus[0].set_xlabel("Mag (V apparent, simulated)")
            ax_legus[0].set_ylim(0, 105)
            ax_legus[0].set_xlim(minmag_legus, maxmag_legus)
            if np.isfinite(lim90):
                ax_legus[0].plot([minmag_legus, lim90], [90, 90], color="#60BD68", linewidth=2, linestyle="--")
                ax_legus[0].text(minmag_legus + 0.07, 91, "90%", fontweight="bold", color="#60BD68")
                ax_legus[0].plot([lim90, lim90], [0, 90], color="#60BD68", linewidth=2, linestyle="--")
            if np.isfinite(lim50):
                ax_legus[0].plot([minmag_legus, lim50], [50, 50], color="#FAA43A", linewidth=2, linestyle="--")
                ax_legus[0].text(minmag_legus + 0.07, 51, "50%", fontweight="bold", color="#FAA43A")
                ax_legus[0].plot([lim50, lim50], [0, 50], color="#FAA43A", linewidth=2, linestyle="--")
            ax_legus[0].grid(True, alpha=0.3)
            # Right: Total recovery after CI cut (our catalogue = CI + ≥4 band merr + M_V ≤ -6)
            ax_legus[1].plot(centers_legus, comp_after_pct, color="grey", linewidth=2)
            ax_legus[1].plot(centers_legus, comp_after_pct, "o", color="grey", markeredgecolor="#4D4D4D")
            ax_legus[1].set_title("Total recovery after CI cut (catalogue)")
            ax_legus[1].set_ylabel("Completeness [%]")
            ax_legus[1].set_xlabel("Mag (V apparent, simulated)")
            ax_legus[1].set_ylim(0, 105)
            ax_legus[1].set_xlim(minmag_legus, maxmag_legus)
            if np.isfinite(lim90_CI):
                ax_legus[1].plot([minmag_legus, lim90_CI], [90, 90], color="#60BD68", linewidth=2, linestyle="--")
                ax_legus[1].text(minmag_legus + 0.07, 91, "90%", fontweight="bold", color="#60BD68")
                ax_legus[1].plot([lim90_CI, lim90_CI], [0, 90], color="#60BD68", linewidth=2, linestyle="--")
            if np.isfinite(lim50_CI):
                ax_legus[1].plot([minmag_legus, lim50_CI], [50, 50], color="#FAA43A", linewidth=2, linestyle="--")
                ax_legus[1].text(minmag_legus + 0.07, 51, "50%", fontweight="bold", color="#FAA43A")
                ax_legus[1].plot([lim50_CI, lim50_CI], [0, 50], color="#FAA43A", linewidth=2, linestyle="--")
            ax_legus[1].grid(True, alpha=0.3)
            fig_legus.suptitle(
                f"LEGUS-style completeness ({GALAXY}, reff={reff_f}, n={n} clusters, {nframe} frames)\n"
                "Right: our catalogue (CI + ≥4 band merr + M_V≤-6) is stricter than original CI-only.",
                fontsize=10,
            )
            plt.tight_layout()
            out_legus = diag_dir / f"completeness_legus_style_{OUTNAME}_reff{reff_f:.2f}.png"
            fig_legus.savefig(out_legus, dpi=120)
            plt.close(fig_legus)
            print(f"Saved LEGUS-style completeness (before/after CI cut) to {out_legus}")

        # Figure-6 style: Recovered completeness [%] vs apparent magnitude [mag], one panel, all bands, 50%/90% ref lines
        fig6, ax6 = plt.subplots(figsize=(7, 5))
        mag_lo, mag_hi = 20.0, 26.0
        colors = ["C0", "C1", "C2", "C3", "C4"]
        markers = ["o", "D", "s", "^", "P"]
        for i in range(n_filters):
            m = mag_5[:, i]
            cx, cy = binned_completeness_mag_range(m, labels, mag_lo=mag_lo, mag_hi=mag_hi, n_bins=30)
            if len(cx) > 0:
                order = np.argsort(cx)
                cx, cy = cx[order], cy[order] * 100.0  # to %
                ax6.plot(cx, cy, "-", color=colors[i], marker=markers[i], markersize=4, label=filter_names[i])
        ax6.axhline(90, color="black", linestyle=":", linewidth=1, label="90%")
        ax6.axhline(50, color="black", linestyle="-.", linewidth=1, label="50%")
        ax6.set_xlim(mag_lo, mag_hi)
        ax6.set_ylim(-5, 105)
        ax6.set_xlabel("Apparent magnitude [mag]")
        ax6.set_ylabel("Completeness [%]")
        ax6.set_title(f"Recovered completeness limits ({GALAXY}, reff={reff_f}, {label_source})")
        ax6.legend(loc="lower left", fontsize=8)
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        out_name6 = f"recovered_completeness_frame0_{OUTNAME}_reff{reff_f:.2f}.png"
        out_path6 = diag_dir / out_name6
        fig6.savefig(out_path6, dpi=150)
        plt.close(fig6)
        print(f"Saved recovered completeness (Figure-6 style) to {out_path6}")

        # Among matched only: denominator = matched in bin (not injected); shows catalogue/M_V turnover at V~24 mag
        if n_matched >= 10 and label_source == "catalogue recovery (in_catalogue)":
            mag_5_m = mag_5[matched_mask]
            lab_m = labels[matched_mask]
            fig_m, ax_m = plt.subplots(figsize=(7, 5))
            for i in range(n_filters):
                m = mag_5_m[:, i]
                cx, cy = binned_completeness_mag_range(m, lab_m, mag_lo=mag_lo, mag_hi=mag_hi, n_bins=30)
                if len(cx) > 0:
                    order = np.argsort(cx)
                    cx, cy = cx[order], cy[order] * 100.0
                    ax_m.plot(cx, cy, "-", color=colors[i], marker=markers[i], markersize=4, label=filter_names[i])
            ax_m.axhline(90, color="black", linestyle=":", linewidth=1, label="90%")
            ax_m.axhline(50, color="black", linestyle="-.", linewidth=1, label="50%")
            ax_m.set_xlim(mag_lo, mag_hi)
            ax_m.set_ylim(-5, 105)
            ax_m.set_xlabel("Apparent magnitude [mag]")
            ax_m.set_ylabel("Completeness [%]")
            ax_m.set_title(f"Among matched only (n={n_matched}); catalogue cuts — M_V=-6 turnover at V~24 mag")
            ax_m.legend(loc="lower left", fontsize=8)
            ax_m.grid(True, alpha=0.3)
            fig_m.tight_layout()
            out_path_m = diag_dir / f"recovered_completeness_matched_only_frame0_{OUTNAME}_reff{reff_f:.2f}.png"
            fig_m.savefig(out_path_m, dpi=150)
            plt.close(fig_m)
            print(f"Saved recovered completeness (matched only) to {out_path_m}")


def main():
    global GALAXY, OUTNAME, DMOD
    parser = argparse.ArgumentParser(description="Pipeline test: injection + stages 2-3 (refactored); optional 4-5 (photometry+CI → binary label)")
    parser.add_argument("--galaxy", type=str, default=GALAXY, help=f"Galaxy/pointing id (default: {GALAXY})")
    parser.add_argument("--outname", type=str, default=OUTNAME, help=f"Output tag (default: {OUTNAME})")
    parser.add_argument("--dmod", type=float, default=DMOD, help=f"Distance modulus (default: {DMOD})")
    parser.add_argument("--sciframe", type=str, default=None, help="White science FITS path (optional override)")
    parser.add_argument(
        "--input_coords", type=str, default=None,
        help="Path to 3-column (x y mag) coordinate file for direct injection",
    )
    parser.add_argument(
        "--no_photometry", action="store_true",
        help="Skip five-filter photometry + CI (Phase B stops at matching). Default is to run photometry + catalogue.",
    )
    parser.add_argument(
        "--nframe", type=int, default=NFRAME,
        help=f"Number of frames (default: {NFRAME})",
    )
    parser.add_argument(
        "--ncl", type=int, default=NCL,
        help=f"Number of clusters per frame when sampling from SLUG (default: {NCL}); ignored if --input_coords is set",
    )
    parser.add_argument(
        "--reff_list", type=str, default=None,
        help="Comma-separated reff values, e.g. '1,2,3,4,5,6,7,8,9,10' (default: single value from ERADIUS=3)",
    )
    parser.add_argument(
        "--sigma_pc", type=float, default=100.0,
        help="Gaussian smoothing for placement footprint in pc (default 100); passed to generate_white_clusters.py",
    )
    parser.add_argument(
        "--placement_mode",
        type=str,
        default="white",
        choices=["white", "uv_mean"],
        help="Cluster placement PDF: white (default) or uv_mean (F275W+F336W mean); passed to generate_white_clusters.py",
    )
    parser.add_argument(
        "--placement_fits",
        type=str,
        default=None,
        help="Optional FITS for placement PDF only; passed to generate_white_clusters.py --placement_fits",
    )
    parser.add_argument(
        "--exclude_region_param",
        nargs="+",
        type=float,
        default=None,
        help="Injection sampling only: CX1 CY1 R1 CX2 CY2 R2 ... (pixels, CX=column CY=row 0-based). "
        "Passed to generate_white_clusters.py.",
    )
    parser.add_argument(
        "--three_panel_region_pix",
        type=str,
        default=None,
        help="Bottom zoom region in image pixels for three-panel recovered plot: 'cx,cy,width,height'. "
        "Example: --three_panel_region_pix 1764,2031,600,350",
    )
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Only backfill physprop (if nframe>1) and run completeness plot; skip injection and pipeline",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove all pipeline outputs (physprop, white/*, tmp, per-filter synthetic) before running",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run Phase B (frame×reff jobs) in parallel. Ignored when running photometry (IRAF is single-process only).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of workers for Phase B when --parallel (default: min(job_count, cpu_count-1))",
    )
    parser.add_argument(
        "--delete_synthetic_after_use", action="store_true",
        help="After each (frame, reff) job, delete the source synthetic FITS and white_position file to free disk (use when storage is limited).",
    )
    parser.add_argument(
        "--run_ml", action="store_true",
        help="After pipeline: run build_ml_inputs then perform_ml_to_learn_completeness (sweep + save best)",
    )
    parser.add_argument(
        "--ml_out_det", type=str, default=None,
        help="Path for build_ml_inputs output det (default: main_dir/det_3d.npy)",
    )
    parser.add_argument(
        "--ml_out_npz", type=str, default=None,
        help="Path for build_ml_inputs output npz (default: main_dir/allprop.npz)",
    )
    parser.add_argument(
        "--ml_out_dir", type=str, default=None,
        help="Output dir for NN sweep and checkpoints (default: <galaxy>/nn_sweep_out)",
    )
    parser.add_argument(
        "--ml_sample_points", type=int, default=None,
        help="Optional sample size for ML quick tests (passed to perform_ml_to_learn_completeness --sample-points).",
    )
    parser.add_argument(
        "--use-white-match", action="store_true",
        help="Pass --use-white-match to build_ml_inputs (white-match labels instead of post-CI)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only run pre-flight path/file check (scripts/check_pipeline_paths.py) and exit. Use before qsub on HPC.",
    )
    parser.add_argument(
        "--skip-galaxy-setup",
        action="store_true",
        help="Do not auto-run setup_legus_galaxy.py when HLSP/white/r2 config are missing under <galaxy>/.",
    )
    cli_args = parser.parse_args()
    GALAXY = cli_args.galaxy.strip()
    OUTNAME = cli_args.outname.strip()
    DMOD = float(cli_args.dmod)
    run_photometry = not cli_args.no_photometry  # default True: always run photometry + catalogue unless --no_photometry

    # Bootstrap LEGUS galaxy directory (download HLSP, white, config) if needed
    if not cli_args.check_only and not cli_args.skip_galaxy_setup:
        ensure_legus_galaxy_setup(GALAXY)

    # Pre-flight check only (guardrail before running / qsub)
    if cli_args.check_only:
        check_script = ROOT / "scripts" / "check_pipeline_paths.py"
        if not check_script.exists():
            print("check_pipeline_paths.py not found.", file=sys.stderr)
            return 1
        cmd = [PYTHON, str(check_script), "--galaxy", GALAXY]
        if run_photometry:
            cmd.append("--run-photometry")
        if not cli_args.skip_galaxy_setup:
            cmd.append("--setup-if-missing")
        return subprocess.run(cmd).returncode

    # Parse reff_list: comma-separated string -> list of floats comma-separated string -> list of floats
    if cli_args.reff_list is not None:
        reff_list = [float(x.strip()) for x in cli_args.reff_list.split(",") if x.strip()]
        print(f"Using reff_list = {reff_list}\n")
    else:
        reff_list = [float(ERADIUS)]
        print(f"Using single reff = {reff_list[0]} (from ERADIUS).\n")

    # When using --input_coords, ncl = number of lines in file; else use --ncl (SLUG sampling)
    if cli_args.input_coords is not None:
        ncl = sum(1 for _ in open(Path(cli_args.input_coords).resolve()) if _.strip())
        print(f"Using ncl={ncl} from --input_coords file.\n")
    else:
        ncl = cli_args.ncl
        print(f"Using ncl={ncl} from SLUG sampling (no --input_coords).\n")
    nframe = cli_args.nframe
    print(f"Using nframe={nframe}.\n")

    bottom_region_pix = None
    if cli_args.three_panel_region_pix:
        try:
            parts = [float(x.strip()) for x in cli_args.three_panel_region_pix.split(",")]
            if len(parts) != 4:
                raise ValueError
            cx, cy, w, h = parts
            if w <= 0 or h <= 0:
                raise ValueError
            bottom_region_pix = (cx, cy, w, h)
            print(
                f"Using three-panel bottom region (pixels): cx={cx}, cy={cy}, width={w}, height={h}\n"
            )
        except ValueError:
            print(
                "ERROR: --three_panel_region_pix must be 'cx,cy,width,height' with width/height > 0",
                file=sys.stderr,
            )
            return 2

    exclude_region_param = cli_args.exclude_region_param
    if exclude_region_param is not None:
        ner = len(exclude_region_param)
        if ner % 3 != 0:
            print(
                "ERROR: --exclude_region_param must have a multiple of 3 values (CX CY R per region).",
                file=sys.stderr,
            )
            return 2
        for i in range(0, ner, 3):
            if exclude_region_param[i + 2] <= 0:
                print(
                    "ERROR: --exclude_region_param: each radius R must be > 0.",
                    file=sys.stderr,
                )
                return 2

    if cli_args.plot_only:
        # Plot-only mode: require existing physprop (from a previous full run).
        # We no longer backfill physprop from white_position here to avoid
        # overwriting SLUG-based 5-band magnitudes or changing the injected sample.
        physprop_dir = ROOT / "physprop"
        missing = []
        for i_frame in range(nframe):
            for reff_f in reff_list:
                base = f"reff{int(reff_f)}_{OUTNAME}"
                for stem in ("mass_select_modelflat", "age_select_modelflat", "mag_VEGA_select_modelflat", "mag_BAO_select_modelflat"):
                    p = physprop_dir / f"{stem}_frame{i_frame}_{base}.npy"
                    if not p.exists():
                        missing.append(str(p))
        if missing:
            print("ERROR: physprop arrays not found for plot_only. Please run scripts/run_pipeline.py "
                  "without --plot_only first (to generate SLUG-based physprop), then rerun with --plot_only.",
                  flush=True)
            return 1
        plot_completeness_diagnostics(nframe=nframe, reff_list=reff_list)
        plot_recovered_on_white_light(
            GALAXY,
            OUTNAME,
            reff_list,
            nframe,
            main_dir=ROOT,
            bottom_region_pix=bottom_region_pix,
        )
        print("\nPlot completed.")
        return 0

    if cli_args.cleanup:
        print("Cleaning all pipeline outputs...")
        cleanup_all_pipeline_outputs()

    sciframe_for_phase_a = ensure_white_sciframe(GALAXY, cli_args.sciframe)
    run_phase_a(
        input_coords=cli_args.input_coords,
        ncl=ncl,
        nframe=nframe,
        reff_list=reff_list,
        sigma_pc=cli_args.sigma_pc,
        sciframe_path=str(sciframe_for_phase_a),
        placement_mode=cli_args.placement_mode,
        placement_fits=cli_args.placement_fits,
        exclude_region_param=exclude_region_param,
    )
    verify_phase_a_outputs(ncl=ncl, nframe=nframe, reff_list=reff_list)
    if run_photometry:
        print("Photometry will inject matched clusters onto HLSP 5-filter science images (same coords as white).", flush=True)
    run_phase_b(ncl=ncl, nframe=nframe, run_photometry=run_photometry, reff_list=reff_list, parallel=cli_args.parallel, n_workers=cli_args.n_workers, delete_synthetic_after_use=cli_args.delete_synthetic_after_use)
    summarise_outputs(ncl=ncl, reff_list=reff_list)
    if run_photometry:
        if cli_args.nframe > 1 and cli_args.input_coords is not None:
            backfill_physprop_from_white_coords(cli_args.nframe, cli_args.input_coords, reff_list=reff_list)
        plot_completeness_diagnostics(nframe=cli_args.nframe, reff_list=reff_list)
        print_catalogue_criterion_counts(nframe=cli_args.nframe, reff_list=reff_list)
    print("\nAll stages completed successfully!")

    # Always build and save ML input artifacts after Phase A+B so downstream jobs
    # can reuse det/npz even when --run_ml is not requested.
    main_dir = ROOT
    galaxy_dir = ROOT / GALAXY
    ml_name = f"{GALAXY}_{OUTNAME}"
    det_path = Path(cli_args.ml_out_det).resolve() if cli_args.ml_out_det else galaxy_dir / f"det_3d_{ml_name}.npy"
    npz_path = Path(cli_args.ml_out_npz).resolve() if cli_args.ml_out_npz else galaxy_dir / f"allprop_{ml_name}.npz"
    ml_out_dir = Path(cli_args.ml_out_dir).resolve() if cli_args.ml_out_dir else (galaxy_dir / "nn_sweep_out")
    ml_out_dir.mkdir(parents=True, exist_ok=True)
    build_cmd = [
        PYTHON, str(ROOT / "scripts" / "build_ml_inputs.py"),
        "--main-dir", str(main_dir),
        "--galaxy", GALAXY,
        "--outname", OUTNAME,
        "--nframe", str(nframe),
        "--reff-list", *[str(r) for r in reff_list],
    ]
    build_cmd += ["--out-det", str(det_path), "--out-npz", str(npz_path)]
    if cli_args.use_white_match:
        build_cmd.append("--use-white-match")
    print("\n--- Running build_ml_inputs (always save det/npz) ---")
    subprocess.run(build_cmd, check=True)
    print(f"Saved ML inputs: det={det_path}, npz={npz_path}")
    # Catalogue recovery on white (in_catalogue from catalogue parquet, not white_match); before heavy cleanup.
    if run_photometry:
        plot_recovered_on_white_light(
            GALAXY,
            OUTNAME,
            reff_list,
            nframe,
            main_dir=ROOT,
            bottom_region_pix=bottom_region_pix,
        )
    print("Cleaning heavy intermediate directories (preserving physprop and ML artifacts)...")
    cleanup_heavy_intermediate_dirs_after_artifacts(GALAXY)

    if cli_args.run_ml:
        ml_cmd = [
            PYTHON, str(ROOT / "scripts" / "perform_ml_to_learn_completeness.py"),
            "--det-path", str(det_path),
            "--npz-path", str(npz_path),
            "--clusters-per-frame", str(ncl),
            "--nframes", str(nframe),
            "--nreff", str(len(reff_list)),
            "--out-dir", str(ml_out_dir),
            "--outname", ml_name,
            "--save-best",
        ]
        if cli_args.ml_sample_points:
            ml_cmd += ["--sample-points", str(cli_args.ml_sample_points)]
        print("\n--- Running perform_ml_to_learn_completeness ---")
        subprocess.run(ml_cmd, check=True)
        print("\nPipeline + ML completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
