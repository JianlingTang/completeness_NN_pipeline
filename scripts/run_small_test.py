#!/usr/bin/env python3
"""
Small end-to-end test of stages 1–3 for ngc628-c (white injection).

Phase A: Run generate_white_clusters.py (legacy injection) with small params.
Phase B: Run refactored pipeline stages 2–3 (detection + matching), optionally
  4–5 (photometry + catalogue). With --run_photometry, matched clusters are
  injected onto HLSP 5-filter science images (same coords as white) via
  scripts/inject_clusters_to_5filters.py; photometry and CI cut run on those
  frames, not on white synthetic images.

Usage:
    python scripts/run_small_test.py                         # SLUG-sampled positions
    python scripts/run_small_test.py --input_coords FILE     # user-supplied positions
"""
import argparse
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
                for name in ["synthetic_fits", "baolab", "synthetic_frames"]:
                    d = sub / name
                    if d.exists():
                        shutil.rmtree(d)
                        print(f"  Removed {d.relative_to(ROOT)}")
    print("Cleanup done.\n")


def run_phase_a(input_coords: str = None, ncl: int = NCL, nframe: int = NFRAME, reff_list: list = None):
    """Phase A: inject synthetic clusters via generate_white_clusters.py."""
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    print("=" * 60)
    print("Phase A: Injection (generate_white_clusters.py)")
    print("=" * 60)

    cmd = [
        PYTHON, str(ROOT / "generate_white_clusters.py"),
        "--ncl", str(ncl),
        "--nframe", str(nframe),
        "--eradius_list",
    ] + [str(int(r)) for r in reff_list] + [
        "--gal_name", GALAXY,
        "--galaxy_fullname", GALAXY,
        "--sciframe", str(ROOT / "ngc628-c" / "ngc628-c_white.fits"),
        "--directory", str(ROOT),
        "--mrmodel", MRMODEL,
        "--dmod", str(DMOD),
        "--outname", OUTNAME,
        "--fits_path", str(ROOT),
        "--psf_path", str(ROOT / "PSF_files"),
        "--bao_path", str(ROOT / ".deps" / "local" / "bin"),
        "--slug_lib_dir", str(ROOT / "SLUG_library"),
    ]
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


def run_phase_b(ncl: int = NCL, nframe: int = NFRAME, run_photometry: bool = False, reff_list: list = None):
    """Phase B: run refactored pipeline stages 2–3 (detection + matching), optionally 4–5 (photometry + catalogue)."""
    if reff_list is None:
        reff_list = [float(ERADIUS)]
    reff_list = [float(r) for r in reff_list]
    print("=" * 60)
    print("Phase B: Detection + Matching" + (" + Photometry + Catalogue" if run_photometry else ""))
    print("=" * 60)
    n_jobs = nframe * len(reff_list)
    print(f"Will process {n_jobs} job(s) (frame 0..{nframe - 1} × reff {reff_list}).")
    if run_photometry:
        print("Photometry: 5 filters per frame, each ~1–2 min — total can be 10+ min per frame.", flush=True)
    print("", flush=True)

    from cluster_pipeline.config import PipelineConfig
    from cluster_pipeline.pipeline import run_galaxy_pipeline

    sex_config = ROOT / "ngc628-c" / f"r2_wl_aa_{GALAXY}.config"
    param_file = ROOT / "output.param"
    nnw_file = ROOT / "default.nnw"

    cfg_kw = dict(
        main_dir=ROOT,
        fits_path=ROOT,
        psf_path=ROOT / "PSF_files",
        bao_path=ROOT / ".deps" / "local" / "bin",
        slug_lib_dir=ROOT / "SLUG_library",
        output_lib_dir=ROOT / "SLUG_library",
        temp_base_dir=ROOT / "tmp_pipeline_test",
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
        keep_frames=True,
    )
    print("Phase B completed successfully.\n")


def summarise_outputs(ncl: int = NCL):
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
    reff_f2 = float(ERADIUS)
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


def plot_completeness_diagnostics(nframe: int = 1, reff_list: list = None):
    """
    Plot completeness (y) vs mass, age, white mag, and mag in 5 bands (x).
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
    filter_names = ["F275W", "F336W", "F435W", "F555W", "F814W"]

    def binned_completeness(x, y_det, n_bins=30):
        ok = np.isfinite(x)
        x_ok, y_ok = x[ok], y_det[ok]
        if len(x_ok) < 2:
            return np.array([]), np.array([]), np.array([])
        bins = np.percentile(x_ok, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.array([np.median(x_ok)]), np.array([np.mean(y_ok)]), np.array([len(x_ok)])
        hist_total, _ = np.histogram(x_ok, bins=bins)
        hist_det, _ = np.histogram(x_ok, bins=bins, weights=y_ok)
        centers = (bins[:-1] + bins[1:]) / 2
        comp = np.where(hist_total > 0, hist_det / hist_total.astype(float), np.nan)
        return centers, comp, hist_total

    def binned_completeness_mag(mag, y_det, n_bins=30):
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
        comp = np.where(hist_total > 0, hist_det / hist_total.astype(float), np.nan)
        return centers, comp, hist_total

    for reff_f in reff_list:
        all_labels = []
        all_mass = []
        all_age = []
        all_mag_white = []
        all_mag5 = []
        label_source = ""

        for i_frame in range(nframe):
            final_labels_path = labels_dir / f"detection_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.npy"
            white_match_path = labels_dir / f"detection_labels_white_match_frame{i_frame}_{OUTNAME}_reff{reff_f:.2f}.npy"
            # Prefer white-match labels so completeness = detection rate (→100% at high mass); use final (after CI) only if no white-match
            if white_match_path.exists():
                lab = np.load(white_match_path)
                if i_frame == 0:
                    label_source = "detection (white-match)"
            elif final_labels_path.exists():
                lab = np.load(final_labels_path)
                if i_frame == 0:
                    label_source = "after photometry+CI"
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
            n_use = min(len(lab), len(mass), len(age), len(mag_5), len(mag_white))
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
            all_mass.append(mass[:n_use])
            all_age.append(age[:n_use])
            all_mag_white.append(mag_white[:n_use])
            all_mag5.append(mag_5[:n_use] if mag_5.ndim >= 2 else mag_5[:n_use].reshape(-1, 1))

        if not all_labels:
            print(f"WARNING: No detection labels/physprop found for reff={reff_f}; skipping completeness plot")
            continue
        labels = np.concatenate(all_labels)
        mass = np.concatenate(all_mass)
        age = np.concatenate(all_age)
        mag_white = np.concatenate(all_mag_white)
        mag_5 = np.concatenate(all_mag5, axis=0)
        if mag_5.ndim == 1:
            mag_5 = mag_5.reshape(-1, 1)
        n = len(labels)
        if n < 10:
            print(f"WARNING: Too few clusters for completeness plot (reff={reff_f}, n={n})")
            continue

        n_filters = min(5, mag_5.shape[1])
        n_cols = 3
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = axes.flatten()

        cx, cy, ct = binned_completeness(np.log10(np.maximum(mass, 1e-10)), labels, n_bins=30)
        ax = axes[0]
        ax.plot(cx, cy, "o-", color="C0")
        ax.set_xlabel(r"$\log_{10}$(mass / M$_\odot$)")
        ax.set_ylabel("Completeness")
        ax.set_title("vs mass")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        cx, cy, ct = binned_completeness(np.log10(np.maximum(age, 1e-7)), labels, n_bins=30)
        ax = axes[1]
        ax.plot(cx, cy, "o-", color="C1")
        ax.set_xlabel(r"$\log_{10}$(age / yr)")
        ax.set_ylabel("Completeness")
        ax.set_title("vs age")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        cx, cy, ct = binned_completeness_mag(mag_white, labels, n_bins=30)
        if len(cx) > 0:
            order = np.argsort(cx)
            cx, cy = cx[order], cy[order]
        ax.plot(cx, cy, "o-", color="C2")
        ax.set_xlabel("mag (white)")
        ax.set_ylabel("Completeness")
        ax.set_title("vs white mag (extraction)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        for i in range(n_filters):
            ax = axes[3 + i]
            m = mag_5[:, i]
            cx, cy, ct = binned_completeness_mag(m, labels, n_bins=30)
            if len(cx) > 0:
                order = np.argsort(cx)
                cx, cy = cx[order], cy[order]
            ax.plot(cx, cy, "o-", color=f"C{(3 + i) % 10}")
            ax.set_xlabel(f"mag ({filter_names[i]})")
            ax.set_ylabel("Completeness")
            ax.set_title(f"vs {filter_names[i]}")
            ax.set_ylim(-0.05, 1.05)
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


def main():
    parser = argparse.ArgumentParser(description="Pipeline test: injection + stages 2-3 (refactored); optional 4-5 (photometry+CI → binary label)")
    parser.add_argument(
        "--input_coords", type=str, default=None,
        help="Path to 3-column (x y mag) coordinate file for direct injection",
    )
    parser.add_argument(
        "--run_photometry", action="store_true",
        help="Run five-filter photometry + CI cut and output final detection_*.npy (requires per-filter synthetic frames)",
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
        "--plot_only", action="store_true",
        help="Only backfill physprop (if nframe>1) and run completeness plot; skip injection and pipeline",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove all pipeline outputs (physprop, white/*, tmp, per-filter synthetic) before running",
    )
    cli_args = parser.parse_args()

    # Parse reff_list: comma-separated string -> list of floats
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

    if cli_args.plot_only:
        if nframe > 1:
            backfill_physprop_from_white_coords(nframe, cli_args.input_coords, reff_list=reff_list)
        plot_completeness_diagnostics(nframe=nframe, reff_list=reff_list)
        print("\nPlot completed.")
        return 0

    if cli_args.cleanup:
        print("Cleaning all pipeline outputs...")
        cleanup_all_pipeline_outputs()

    run_phase_a(input_coords=cli_args.input_coords, ncl=ncl, nframe=nframe, reff_list=reff_list)
    verify_phase_a_outputs(ncl=ncl, nframe=nframe, reff_list=reff_list)
    if cli_args.run_photometry:
        print("Photometry will inject matched clusters onto HLSP 5-filter science images (same coords as white).", flush=True)
    run_phase_b(ncl=ncl, nframe=nframe, run_photometry=cli_args.run_photometry, reff_list=reff_list)
    summarise_outputs(ncl=ncl)
    if cli_args.run_photometry:
        if cli_args.nframe > 1 and cli_args.input_coords is not None:
            backfill_physprop_from_white_coords(cli_args.nframe, cli_args.input_coords, reff_list=reff_list)
        plot_completeness_diagnostics(nframe=cli_args.nframe, reff_list=reff_list)
    print("\nAll stages completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
