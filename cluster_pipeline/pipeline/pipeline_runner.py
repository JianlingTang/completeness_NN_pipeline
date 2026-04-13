"""
Pipeline orchestrator: run the full galaxy pipeline (or a subset) with clear stages.
Uses config only; no global state or os.chdir. Designed for HPC: each (galaxy, frame, reff)
can run in isolation with its own temp dir.

Stages (use max_stage to run only the first N):
  1 = injection    copy/generate synthetic frame + coords
  2 = detection   SExtractor on frame
  3 = matching   match injected vs detected coords; write matched_coords
  4 = photometry (optional)  five-filter aperture photometry using matched_coords
  5 = catalogue  CI/merr cuts; in_catalogue; then binary label detection_*.npy
  6 = dataset    build final dataset parquet/npy
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..catalogue.catalogue_filters import B_FILTER, I_FILTER, VBAND_FILTER
from ..config import PipelineConfig, get_config
from ..data.models import DetectionResult, MatchResult
from ..detection import SExtractorRunner
from ..matching import CoordinateMatcher, load_coords_white_position
from ..utils.filesystem import ensure_dir, safe_remove_tree
from ..utils.logging_utils import get_logger
from .diagnostics import write_match_summary
from .injection_5filter import write_matched_coords_per_filter
from .stages import (
    STAGE_CATALOGUE,
    STAGE_DETECTION,
    STAGE_INJECTION,
    STAGE_MATCHING,
    STAGE_NAMES,
    STAGE_PHOTOMETRY,
    run_stage,
)

logger = get_logger(__name__)


def _delete_synthetic_for_job(
    config: "PipelineConfig",
    galaxy_id: str,
    frame_id: int,
    reff: float,
    outname: str,
) -> None:
    """
    Remove source synthetic FITS and white_position file for this (frame_id, reff)
    to free disk after Phase B has finished. Call only after the job succeeded.
    """
    synth_dir = config.synthetic_fits_dir(galaxy_id)
    white_dir = config.white_dir(galaxy_id)
    pattern = f"*_frame{frame_id}_{outname}_reff{reff:.2f}.fits"
    removed = 0
    for p in synth_dir.glob(pattern):
        try:
            p.unlink()
            removed += 1
            logger.debug("Deleted synthetic FITS %s", p.name)
        except OSError as e:
            logger.warning("Could not remove %s: %s", p, e)
    coord_path = white_dir / f"white_position_{frame_id}_{outname}_reff{reff:.2f}.txt"
    if coord_path.exists():
        try:
            coord_path.unlink()
            removed += 1
            logger.debug("Deleted coords %s", coord_path.name)
        except OSError as e:
            logger.warning("Could not remove %s: %s", coord_path, e)
    if removed:
        logger.info("Freed storage: removed %s file(s) for frame_id=%s reff=%s", removed, frame_id, reff)


def _run_one_frame_reff_worker(args: tuple) -> None:
    """
    Worker for parallel Phase B: run one (frame_id, reff) in isolation.
    Creates its own matcher and sextractor to avoid shared state; each job writes
    to distinct paths (temp_dir, matched_coords, detection_labels, etc.) so no file conflict.
    """
    (
        galaxy_id,
        frame_id,
        reff,
        outname,
        config,
        run_injection,
        run_detection,
        run_matching,
        run_photometry,
        run_catalogue,
        keep_frames,
        delete_synthetic_after_use,
    ) = args
    temp_dir = config.temp_dir_for(galaxy_id, frame_id, reff)
    matcher = CoordinateMatcher(tolerance_pix=config.thres_coord)
    sextractor = SExtractorRunner(config)
    try:
        _run_one_frame_reff(
            galaxy_id=galaxy_id,
            frame_id=frame_id,
            reff=reff,
            outname=outname,
            config=config,
            temp_dir=temp_dir,
            matcher=matcher,
            sextractor=sextractor,
            run_injection=run_injection,
            run_detection=run_detection,
            run_matching=run_matching,
            run_photometry=run_photometry,
            run_catalogue=run_catalogue,
            keep_frames=keep_frames,
        )
        if delete_synthetic_after_use:
            _delete_synthetic_for_job(config, galaxy_id, frame_id, reff, outname)
    finally:
        if not keep_frames and temp_dir.exists():
            safe_remove_tree(temp_dir)
            logger.debug("Removed temp dir %s", temp_dir)


def run_galaxy_pipeline(
    galaxy_id: str,
    config: PipelineConfig | None = None,
    *,
    outname: str = "pipeline",
    max_stage: int | None = None,
    run_injection: bool = True,
    run_detection: bool = True,
    run_matching: bool = True,
    run_photometry: bool = False,
    run_catalogue: bool = False,
    keep_frames: bool = False,
    delete_synthetic_after_use: bool = False,
    parallel: bool = False,
    n_workers: int | None = None,
) -> None:
    """
    Run the completeness pipeline for one galaxy.

    Stages (use max_stage to run only the first N):
      1 = injection    copy/generate synthetic frame + coords
      2 = detection    SExtractor on frame
      3 = matching     match injected vs detected coords; write matched_coords
      4 = photometry   (optional)
      5 = catalogue    CI/merr; in_catalogue
      6 = dataset      build final dataset parquet/npy

    Parameters
    ----------
    galaxy_id : str
        Galaxy identifier (e.g. "ngc628-c_white-R17v100").
    config : PipelineConfig, optional
        If None, uses get_config().
    outname : str
        Output run name (for filenames).
    max_stage : int, optional
        If set, run only stages 1..max_stage (e.g. max_stage=2 → injection + detection).
        If None, use individual run_* flags.
    run_injection, run_detection, run_matching : bool
        Used when max_stage is None.
    run_photometry : bool
        Run photometry stage (five-filter, uses matched_coords).
    run_catalogue : bool
        Run catalogue (CI/merr) and write final detection_*.npy from in_catalogue.
    keep_frames : bool
        If False, synthetic frames are deleted after detection + match.
    delete_synthetic_after_use : bool
        If True, after each (frame, reff) job completes, remove the source synthetic FITS
        and white_position file for that job to free disk (recommended when storage is limited).
    parallel : bool
        If True and more than one (frame, reff) job, run jobs in a process pool (no shared state; each job uses its own temp dir and output paths).
    n_workers : int, optional
        Number of worker processes when parallel=True. If None, uses min(job_count, cpu_count - 1).
    """
    cfg = config or get_config()
    # Allow parallel + photometry for this test run
    # if run_photometry and parallel:
    #     logger.info("run_photometry=True: disabling parallel (IRAF single-process only)")
    #     parallel = False
    if max_stage is not None:
        run_injection = run_stage(STAGE_INJECTION, max_stage)
        run_detection = run_stage(STAGE_DETECTION, max_stage)
        run_matching = run_stage(STAGE_MATCHING, max_stage)
        run_photometry = run_photometry or run_stage(STAGE_PHOTOMETRY, max_stage)
        run_catalogue = run_catalogue or run_stage(STAGE_CATALOGUE, max_stage)
        logger.info(
            "Pipeline galaxy=%s outname=%s max_stage=%s (stages: %s)",
            galaxy_id, outname, max_stage,
            [STAGE_NAMES[s] for s in range(1, max_stage + 1) if s in STAGE_NAMES],
        )
    else:
        logger.info("Starting pipeline for galaxy=%s outname=%s", galaxy_id, outname)

    matcher = CoordinateMatcher(tolerance_pix=cfg.thres_coord)
    sextractor = SExtractorRunner(cfg)

    # Ensure base dirs exist
    ensure_dir(cfg.temp_base_dir)
    labels_dir = cfg.matched_coords_dir(galaxy_id)
    ensure_dir(labels_dir)

    nframe = cfg.nframe
    reff_list = cfg.reff_list

    total_jobs = len(reff_list) * nframe
    jobs = [(frame_id, reff) for frame_id in range(nframe) for reff in reff_list]

    if parallel and total_jobs > 1:
        import multiprocessing as mp
        workers = n_workers if n_workers is not None else min(total_jobs, max(1, mp.cpu_count() - 1))
        logger.info("Phase B parallel: %s workers, %s jobs", workers, total_jobs)
        args_list = [
            (
                galaxy_id,
                frame_id,
                reff,
                outname,
                cfg,
                run_injection,
                run_detection,
                run_matching,
                run_photometry,
                run_catalogue,
                keep_frames,
                delete_synthetic_after_use,
            )
            for (frame_id, reff) in jobs
        ]
        with mp.Pool(processes=workers) as pool:
            pool.map(_run_one_frame_reff_worker, args_list)
        logger.info("Pipeline finished for galaxy=%s", galaxy_id)
        return

    # Sequential: one matcher/sextractor shared across jobs
    matcher = CoordinateMatcher(tolerance_pix=cfg.thres_coord)
    sextractor = SExtractorRunner(cfg)
    job_idx = 0
    for frame_id in range(nframe):
        for reff in reff_list:
            job_idx += 1
            print(f"[Phase B] Job {job_idx}/{total_jobs}: frame_id={frame_id} reff={reff:.2f}", flush=True)
            temp_dir = cfg.temp_dir_for(galaxy_id, frame_id, reff)
            try:
                _run_one_frame_reff(
                    galaxy_id=galaxy_id,
                    frame_id=frame_id,
                    reff=reff,
                    outname=outname,
                    config=cfg,
                    temp_dir=temp_dir,
                    matcher=matcher,
                    sextractor=sextractor,
                    run_injection=run_injection,
                    run_detection=run_detection,
                    run_matching=run_matching,
                    run_photometry=run_photometry,
                    run_catalogue=run_catalogue,
                    keep_frames=keep_frames,
                )
                if delete_synthetic_after_use:
                    _delete_synthetic_for_job(cfg, galaxy_id, frame_id, reff, outname)
            except Exception as e:
                logger.exception(
                    "Failed galaxy=%s frame=%s reff=%s: %s",
                    galaxy_id, frame_id, reff, e,
                )
                raise
            finally:
                if not keep_frames and temp_dir.exists():
                    safe_remove_tree(temp_dir)
                    logger.debug("Removed temp dir %s", temp_dir)

    logger.info("Pipeline finished for galaxy=%s", galaxy_id)


def _run_one_frame_reff(
    galaxy_id: str,
    frame_id: int,
    reff: float,
    outname: str,
    config: PipelineConfig,
    temp_dir: Path,
    matcher: CoordinateMatcher,
    sextractor: SExtractorRunner,
    run_injection: bool,
    run_detection: bool,
    run_matching: bool,
    run_photometry: bool,
    run_catalogue: bool,
    keep_frames: bool,
) -> None:
    """
    Run injection → detection → matching for one (frame, reff).
    Frame and catalogs live in temp_dir; caller deletes temp_dir if keep_frames is False.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: in production, FrameInjector writes frame_path and coord_path into temp_dir.
    # Here we optionally use existing synthetic_fits for testing without running injection.
    frame_path = temp_dir / "injected.fits"
    coord_path = temp_dir / "injected_coords.txt"
    if run_injection:
        print(f"  [Phase B] Copying frame {frame_id} and coords to temp dir...", flush=True)
        synthetic_fits = config.synthetic_fits_dir(galaxy_id)
        white_dir = config.white_dir(galaxy_id)
        frame_pattern = f"*_frame{frame_id}_{outname}_reff{reff:.2f}.fits"
        coord_name = f"white_position_{frame_id}_{outname}_reff{reff:.2f}.txt"
        existing_frames = list(synthetic_fits.glob(frame_pattern))
        existing_coord = white_dir / coord_name
        if existing_frames and existing_coord.exists():
            import shutil
            shutil.copy(existing_frames[0], frame_path)
            shutil.copy(existing_coord, coord_path)
        else:
            logger.warning(
                "Injection requested but no existing frame/coords for frame_id=%s reff=%s",
                frame_id, reff,
            )
            return

    if not frame_path.exists() or not coord_path.exists():
        logger.warning("Skipping frame_id=%s reff=%s: missing frame or coords", frame_id, reff)
        return

    if run_detection:
        print(f"  [Phase B] Running SExtractor on frame {frame_id} (can take 1–5 min for large FITS)...", flush=True)
        det: DetectionResult = sextractor.run(
            frame_path=frame_path,
            output_dir=temp_dir,
            catalog_name=f"det_frame{frame_id}_reff{reff:.2f}.cat",
            coo_suffix=".coo",
        )
        logger.info("Detection frame_id=%s reff=%s n_detected=%s", frame_id, reff, det.n_detected)
        print(f"  [Phase B] SExtractor: n_detected={det.n_detected}", flush=True)
    else:
        coo_path = temp_dir / f"{frame_path.stem}.coo"
        if not coo_path.exists():
            logger.warning("No detection run and no .coo file; skipping match")
            return
        det = DetectionResult(
            catalog_path=temp_dir / "det.cat",
            coord_path=coo_path,
            n_detected=0,
            frame_path=frame_path,
        )

    out_matched = config.matched_coords_dir(galaxy_id) / (
        f"matched_frame{frame_id}_{outname}_reff{reff:.2f}.txt"
    )
    match_results_path = config.catalogue_dir(galaxy_id) / (
        f"match_results_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    )
    labels_dir = config.detection_labels_dir(galaxy_id)
    ensure_dir(labels_dir)

    if run_matching:
        # white_position files are (x y mag); load_coords_white_position returns (x, y) = (col, row)
        injected_xy = load_coords_white_position(coord_path)
        # SExtractor uses 1-based pixel coords (FITS); injector file is 0-based (x, y)
        injected_xy = injected_xy + 1.0
        n_injected = len(injected_xy)
        cluster_ids = list(range(n_injected))  # Stable IDs when injector does not provide them
        match_result: MatchResult = matcher.match(
            injected_xy, det.coord_path, cluster_ids=cluster_ids
        )
        logger.info(
            "Match frame_id=%s reff=%s n_matched=%s/%s",
            frame_id, reff, match_result.n_matched, match_result.n_injected,
        )
        print(f"  [Phase B] Match: n_matched={match_result.n_matched}/{match_result.n_injected}", flush=True)
        matcher.write_matched_coords(
            match_result, out_matched, det.coord_path,
            # Always persist cluster_ids so downstream serial photometry pass can reuse
            # parallel matching outputs safely.
            include_cluster_id=True,
        )
        # Write match_results parquet (all cluster_ids, matched 0/1) for build_final_detection order
        diag_dir = config.diagnostics_dir(galaxy_id)
        ensure_dir(diag_dir)
        summary_path = diag_dir / f"match_summary_frame{frame_id}_reff{reff:.2f}_{outname}.txt"
        write_match_summary(coord_path, match_result, summary_path)
        ensure_dir(match_results_path.parent)
        match_df = _build_match_results_df(
            match_result, galaxy_id, frame_id, reff, injected_xy
        )
        match_df.to_parquet(match_results_path, index=False)
        logger.info("Wrote match_results to %s", match_results_path)
    else:
        # Reuse precomputed matching outputs (useful when doing photometry serial after parallel stages 1-3).
        if not out_matched.exists() or not match_results_path.exists():
            raise RuntimeError(
                "run_matching=False but required matched outputs are missing: "
                f"{out_matched}, {match_results_path} (frame={frame_id}, reff={reff:.2f})"
            )
        match_df = pd.read_parquet(match_results_path)
        if "cluster_id" not in match_df.columns or "matched" not in match_df.columns:
            raise ValueError(
                f"Malformed match_results parquet: {match_results_path} (needs cluster_id, matched)"
            )
        cluster_ids = [int(x) for x in match_df["cluster_id"].tolist()]
        matched_flags = [bool(x) for x in match_df["matched"].tolist()]
        matched_indices = [i for i, m in enumerate(matched_flags) if m]
        matched_positions: list[tuple[float, float]] = []
        if {"detected_x", "detected_y"} <= set(match_df.columns):
            for i in matched_indices:
                dx = match_df.iloc[i]["detected_x"]
                dy = match_df.iloc[i]["detected_y"]
                if pd.notna(dx) and pd.notna(dy):
                    matched_positions.append((float(dx), float(dy)))
        match_result = MatchResult(
            injected_path=coord_path,
            detected_path=det.coord_path,
            cluster_ids=cluster_ids,
            matched_indices=matched_indices,
            matched_positions=matched_positions,
            n_injected=len(cluster_ids),
            n_matched=len(matched_indices),
            tolerance_pix=float(config.thres_coord),
        )
        print(
            f"  [Phase B] Reusing matching outputs (frame {frame_id}, reff={reff:.2f}); "
            "skip matching, run photometry+CI serial.",
            flush=True,
        )
        logger.info(
            "Reused match_results frame_id=%s reff=%s n_matched=%s/%s",
            frame_id, reff, match_result.n_matched, match_result.n_injected,
        )

    # Binary label: from catalogue (after photometry+CI) if running stages 4–5, else white-match only
    if run_photometry and run_catalogue:
        try:
            # Optionally inject matched clusters onto HLSP science images (same coords on all 5 filters)
            inject_script = getattr(config, "inject_5filter_script", None)
            if inject_script is not None and match_result.n_matched > 0:
                from ..data.galaxy_metadata import GalaxyMetadata
                meta = GalaxyMetadata.load(config.main_dir, galaxy_id, config.fits_path)
                filters = getattr(meta, "filters", [])
                if len(filters) >= 1:
                    mag_vega_path = config.physprop_dir() / (
                        f"mag_VEGA_select_model{config.mrmodel}_frame{frame_id}_reff{int(reff):d}_{outname}.npy"
                    )
                    cluster_ids_path = out_matched.parent / (out_matched.stem + "_cluster_ids.txt")
                    if mag_vega_path.exists() and cluster_ids_path.exists():
                        write_matched_coords_per_filter(
                            matched_coords_path=out_matched,
                            cluster_ids_path=cluster_ids_path,
                            mag_vega_path=mag_vega_path,
                            white_dir=config.white_dir(galaxy_id),
                            frame_id=frame_id,
                            reff=reff,
                            outname=outname,
                            filter_names=filters[:5],
                        )
                        print(f"  [Phase B] Injecting matched clusters onto HLSP 5-filter images (frame {frame_id})...", flush=True)
                        cmd = [
                            sys.executable,
                            str(Path(inject_script).resolve()),
                            "--directory", str(config.main_dir),
                            "--fits-dir", str(config.fits_path),
                            "--gal_name", galaxy_id.split("_")[0],
                            "--outname", outname,
                            "--eradius_list", str(int(reff)),
                            "--nframe_start", str(frame_id),
                            "--nframe_end", str(frame_id + 1),
                            "--nfilter_start", "0",
                            "--nfilter_end", str(min(5, len(filters))),
                            "--use_white",
                            "--ncl", str(match_result.n_matched),
                            "--dmod", str(config.dmod),
                            "--mrmodel", config.mrmodel,
                        ]
                        if getattr(config, "overwrite", False):
                            cmd.append("--overwrite")
                        subprocess.run(cmd, cwd=str(config.main_dir), check=True)
                    else:
                        raise RuntimeError(
                            "physprop mag_VEGA or cluster_ids missing for 5-filter injection. "
                            f"Required paths: {mag_vega_path}, {cluster_ids_path}"
                        )
            print(f"  [Phase B] Running photometry + CI cut (frame {frame_id}: 5 filters, may take 1–2 min each)...", flush=True)
            _run_photometry_and_catalogue(
                galaxy_id=galaxy_id,
                frame_id=frame_id,
                reff=reff,
                outname=outname,
                config=config,
                matched_coords_path=out_matched,
                match_result=match_result,
                match_results_path=match_results_path,
            )
            from ..catalogue import build_final_detection, save_final_detection
            cat_path = config.catalogue_dir(galaxy_id) / (
                f"catalogue_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
            )
            if cat_path.exists():
                labels = build_final_detection(cat_path, match_results_path)
                labels_path = labels_dir / f"detection_frame{frame_id}_{outname}_reff{reff:.2f}.npy"
                save_final_detection(labels, labels_path)
                logger.info("Wrote final binary labels (after photometry+CI) to %s", labels_path)
                # Also write white-match labels so completeness vs mass can show detection rate (→100% at high mass)
                _write_white_match_labels(match_result, labels_dir, frame_id, outname, reff)
            else:
                raise RuntimeError(
                    f"Catalogue parquet not found after photometry/CI: {cat_path}"
                )
        except Exception as e:
            logger.exception("Photometry/catalogue failed: %s", e)
            raise
    else:
        _write_white_match_labels(match_result, labels_dir, frame_id, outname, reff)


def _build_match_results_df(
    match_result: MatchResult,
    galaxy_id: str,
    frame_id: int,
    reff: float,
    injected_xy: np.ndarray,
) -> pd.DataFrame:
    """Build match_results parquet: one row per cluster_id with matched 0/1 and coords."""
    rows = []
    injected = np.atleast_2d(injected_xy)
    matched_set = set(match_result.matched_indices)
    for i, cid in enumerate(match_result.cluster_ids):
        m = 1 if i in matched_set else 0
        ix = float(injected[i, 0]) if i < len(injected) else 0.0
        iy = float(injected[i, 1]) if i < len(injected) else 0.0
        dx = dy = np.nan
        if m and match_result.matched_indices:
            try:
                pos_idx = match_result.matched_indices.index(i)
            except ValueError:
                pos_idx = 0
            if pos_idx < len(match_result.matched_positions):
                dx, dy = match_result.matched_positions[pos_idx]
        rows.append({
            "cluster_id": cid,
            "galaxy_id": galaxy_id,
            "frame_id": frame_id,
            "reff": reff,
            "matched": m,
            "injected_x": ix,
            "injected_y": iy,
            "detected_x": dx,
            "detected_y": dy,
        })
    return pd.DataFrame(rows)


def _write_white_match_labels(
    match_result: MatchResult,
    labels_dir: Path,
    frame_id: int,
    outname: str,
    reff: float,
) -> None:
    """Write white-match-only binary labels (before photometry/CI)."""
    path = labels_dir / f"detection_labels_white_match_frame{frame_id}_{outname}_reff{reff:.2f}.npy"
    np.save(path, np.array(match_result.detection_labels, dtype=np.uint8))
    logger.info("Wrote white-match labels to %s", path)


def _load_aperture_corrections(config: "PipelineConfig", galaxy_id: str) -> dict[str, tuple[float, float]]:
    """
    Load avg_aperture_correction_{gal}.txt (same path and format as original_photometry_on_5_and_CI.py).
    File: 3 columns (filter, ap, aperr) dtype str. Returns dict filter_name -> (apcorr, apcorrerr).
    Raises FileNotFoundError if the file is missing (required for photometry).
    """
    gal_short = galaxy_id.split("_")[0]
    fname = f"avg_aperture_correction_{galaxy_id}.txt"
    bases = [config.main_dir / galaxy_id]
    if getattr(config, "fits_path", None) is not None:
        bases.append(config.fits_path / galaxy_id)
    path = None
    for base in bases:
        p = base / fname
        if p.exists():
            path = p
            break
        p_alt = base / f"avg_aperture_correction_{gal_short}.txt"
        if p_alt.exists():
            path = p_alt
            break
    if path is None:
        tried = [str(b / fname) for b in bases] + [str(b / f"avg_aperture_correction_{gal_short}.txt") for b in bases]
        raise FileNotFoundError(
            f"avg_aperture_correction_{galaxy_id}.txt (or avg_aperture_correction_{gal_short}.txt) not found. "
            f"Tried: {tried}. Photometry requires this file (same as original_photometry_on_5_and_CI.py)."
        )
    out = {}
    data = np.loadtxt(path, usecols=(0, 1, 2), dtype="str", ndmin=2)
    for row in data:
        filt_key = str(row[0]).strip()
        apcorr = float(row[1])
        apcorrerr = float(row[2])
        out[filt_key] = (apcorr, apcorrerr)
        out[filt_key.upper()] = (apcorr, apcorrerr)
        out[filt_key.lower()] = (apcorr, apcorrerr)
    logger.info("Loaded aperture corrections from %s (%s filters)", path, len(data))
    return out


def _run_photometry_and_catalogue(
    galaxy_id: str,
    frame_id: int,
    reff: float,
    outname: str,
    config: PipelineConfig,
    matched_coords_path: Path,
    match_result: MatchResult,
    match_results_path: Path,
) -> None:
    """
    Run five-filter aperture photometry (using matched_coords), CI cut, build photometry parquet,
    apply catalogue filters, write catalogue parquet. Flow and formulae match original_photometry_on_5_and_CI.py
    (aperture correction from avg_aperture_correction_*.txt, then merr/CI cut).
    """
    from ..catalogue import apply_catalogue_filters, write_catalogue_parquet
    from ..data.galaxy_metadata import GalaxyMetadata
    from ..photometry.aperture_photometry import run_aperture_photometry
    from ..photometry.ci_filter import apply_ci_cut

    ensure_dir(config.catalogue_dir(galaxy_id))
    try:
        meta = GalaxyMetadata.load(config.main_dir, galaxy_id, config.fits_path)
    except Exception as e:
        raise RuntimeError(f"Could not load galaxy metadata for photometry: {e}") from e
    filters = getattr(meta, "filters", [])
    if not filters:
        raise RuntimeError("No filters in galaxy metadata; cannot run photometry.")
    ci_threshold = getattr(meta, "ci_cut", None) or 1.4  # default 1.4 (source of truth: perform_photometry_ci_cut_on_5filters.py)
    merr_cut = config.merr_cut
    # Aperture corrections per filter (same as original_photometry_on_5_and_CI.py)
    aperture_corrections = _load_aperture_corrections(config, galaxy_id)
    # Canonical 5-band: 275, 336, 435, 555, 814; criteria use V=F555W, B=F435W, I=F814W
    vband_name = VBAND_FILTER if VBAND_FILTER in filters else (next((f for f in filters if "555" in f or "F555" in f), filters[0]) if filters else "F555W")

    frame_pattern = f"*_frame{frame_id}_{outname}_reff{reff:.2f}.fits"
    cluster_ids_path = matched_coords_path.parent / (matched_coords_path.stem + "_cluster_ids.txt")
    if not cluster_ids_path.exists():
        cluster_ids = match_result.get_matched_cluster_ids()
    else:
        cluster_ids = [int(x) for x in cluster_ids_path.read_text().splitlines() if x.strip()]

    phot_rows: list[dict] = []
    for filt in filters:
        print(f"    [Photometry] frame {frame_id} filter {filt}...", flush=True)
        synth_dir = config.filter_synthetic_fits_dir(galaxy_id, filt)
        frames = list(synth_dir.glob(frame_pattern))
        if not frames:
            raise RuntimeError(
                f"No synthetic frame for filter {filt} frame={frame_id} reff={reff:.2f} "
                f"using pattern {frame_pattern} in {synth_dir}"
            )
        frame_path = frames[0]
        out_phot = config.photometry_dir(galaxy_id, filt)
        ensure_dir(out_phot)
        try:
            zp = meta.zeropoints.get(filt, 0.0)
            if filt not in meta.exptimes:
                raise KeyError(
                    f"Filter '{filt}' has no exptime in header_info. "
                    "header_info must have 4 columns (filter, instrument, zeropoint, exptime) for all filters."
                )
            exptime = meta.exptimes[filt]
            ap = meta.aperture_radius or 3.0
            apertures = [1.0, 3.0] if ap in (1.0, 3.0) else [1.0, 3.0, ap]
            print(f"      zeropoint={zp}, EXPTIME={exptime}", flush=True)
            mag_txt = run_aperture_photometry(
                frame_path=frame_path,
                coords_path=matched_coords_path,
                output_dir=out_phot,
                filter_name=filt,
                zeropoint=zp,
                exptime=exptime,
                aperture_radii=apertures,
                user_aperture=ap,
                photometry_dir=out_phot,
            )
            if aperture_corrections:
                apcorr, apcorrerr = aperture_corrections.get(filt, aperture_corrections.get(filt.upper(), (0.0, 0.0)))
            else:
                apcorr, apcorrerr = None, None
            (mag_4, merr_4, ci_values), (passes_ci, passes_merr, keep) = apply_ci_cut(
                mag_txt, ci_threshold, merr_cut,
                filter_is_vband=("555" in filt or "F555" in filt),
                user_aperture=ap,
                apcorr=apcorr,
                apcorrerr=apcorrerr,
            )
            n_rows = len(mag_4)
            mag_1px = np.full(n_rows, np.nan)
            mag_3px = np.full(n_rows, np.nan)
            for k in range(n_rows):
                mag_3px[k] = mag_4[k]
                mag_1px[k] = mag_4[k] + ci_values[k]
            for k in range(n_rows):
                cid = cluster_ids[k] if k < len(cluster_ids) else k
                phot_rows.append({
                    "cluster_id": cid,
                    "galaxy_id": galaxy_id,
                    "frame_id": frame_id,
                    "reff": reff,
                    "filter_name": filt,
                    "mag": float(mag_4[k]),
                    "merr": float(merr_4[k]),
                    "mag_1px": float(mag_1px[k]),
                    "mag_3px": float(mag_3px[k]),
                    "ci": float(ci_values[k]),
                    "passes_ci": int(passes_ci[k]) if k < len(passes_ci) else 0,
                    "passes_merr": int(passes_merr[k]) if k < len(passes_merr) else 0,
                })
        except Exception as e:
            raise RuntimeError(f"Photometry failed for filter {filt}: {e}") from e

    if not phot_rows:
        raise RuntimeError("No photometry rows produced; catalogue cannot be built.")
    phot_df = pd.DataFrame(phot_rows)
    phot_path = config.catalogue_dir(galaxy_id) / (
        f"photometry_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    )
    phot_df.to_parquet(phot_path, index=False)
    logger.info("Wrote photometry parquet to %s", phot_path)
    dmod = getattr(meta, "distance_modulus", None) or config.dmod
    cat_df = apply_catalogue_filters(
        phot_path,
        merr_cut=merr_cut,
        vband_filter=vband_name,
        b_filter=B_FILTER,
        i_filter=I_FILTER,
        dmod=dmod,
    )
    cat_path = config.catalogue_dir(galaxy_id) / (
        f"catalogue_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    )
    write_catalogue_parquet(cat_df, cat_path)
    logger.info("Wrote catalogue parquet to %s", cat_path)
