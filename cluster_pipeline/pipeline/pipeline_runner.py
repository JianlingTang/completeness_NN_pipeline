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
    """
    cfg = config or get_config()
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

    if run_matching:
        # white_position files are (y x mag); convert to (x, y) for SExtractor/matcher
        injected_xy = load_coords_white_position(coord_path)
        # SExtractor uses 1-based pixel coords (FITS); injector file is 0-based (row, col)
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
        # Write matched coords to persistent dir for photometry if needed
        out_matched = config.matched_coords_dir(galaxy_id) / (
            f"matched_frame{frame_id}_{outname}_reff{reff:.2f}.txt"
        )
        matcher.write_matched_coords(
            match_result, out_matched, det.coord_path,
            include_cluster_id=(run_photometry or run_catalogue),
        )
        # Write match_results parquet (all cluster_ids, matched 0/1) for build_final_detection order
        diag_dir = config.diagnostics_dir(galaxy_id)
        ensure_dir(diag_dir)
        summary_path = diag_dir / f"match_summary_frame{frame_id}_reff{reff:.2f}_{outname}.txt"
        write_match_summary(coord_path, match_result, summary_path)
        match_results_path = config.catalogue_dir(galaxy_id) / (
            f"match_results_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
        )
        ensure_dir(match_results_path.parent)
        match_df = _build_match_results_df(
            match_result, galaxy_id, frame_id, reff, injected_xy
        )
        match_df.to_parquet(match_results_path, index=False)
        logger.info("Wrote match_results to %s", match_results_path)
        # Binary label: from catalogue (after photometry+CI) if running stages 4–5, else white-match only
        labels_dir = config.detection_labels_dir(galaxy_id)
        ensure_dir(labels_dir)
        if run_photometry and run_catalogue:
            try:
                # Optionally inject matched clusters onto HLSP science images (same coords on all 5 filters)
                inject_script = getattr(config, "inject_5filter_script", None)
                if inject_script is not None and match_result.n_matched > 0:
                    from ..data.galaxy_metadata import GalaxyMetadata
                    meta = GalaxyMetadata.load(config.main_dir, galaxy_id)
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
                                "--galaxy_fullname", galaxy_id,
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
                            logger.warning(
                                "physprop mag_VEGA or cluster_ids missing; skip inject. Paths: %s, %s",
                                mag_vega_path, cluster_ids_path,
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
                    logger.warning("Catalogue parquet not found; writing white-match labels only")
                    _write_white_match_labels(match_result, labels_dir, frame_id, outname, reff)
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
    apply catalogue filters, write catalogue parquet. Does not write detection_*.npy (caller does).
    """
    from ..catalogue import apply_catalogue_filters, write_catalogue_parquet
    from ..data.galaxy_metadata import GalaxyMetadata
    from ..photometry.aperture_photometry import run_aperture_photometry
    from ..photometry.ci_filter import apply_ci_cut

    ensure_dir(config.catalogue_dir(galaxy_id))
    try:
        meta = GalaxyMetadata.load(config.main_dir, galaxy_id)
    except Exception as e:
        logger.warning("Could not load galaxy metadata for photometry: %s", e)
        return
    filters = getattr(meta, "filters", [])
    if not filters:
        logger.warning("No filters in galaxy metadata; skipping photometry")
        return
    ci_threshold = getattr(meta, "ci_cut", None) or 1.4  # default 1.4 (source of truth: perform_photometry_ci_cut_on_5filters.py)
    merr_cut = config.merr_cut
    vband_name = "F555W"
    if vband_name not in filters:
        vband_name = next((f for f in filters if "555" in f or "F555" in f), filters[0] if filters else "")

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
            logger.debug("No synthetic frame for filter %s; skipping", filt)
            continue
        frame_path = frames[0]
        out_phot = config.photometry_dir(galaxy_id, filt)
        ensure_dir(out_phot)
        try:
            zp = meta.zeropoints.get(filt, 0.0)
            ap = meta.aperture_radius or 3.0
            apertures = [1.0, 3.0] if ap in (1.0, 3.0) else [1.0, 3.0, ap]
            exptime = 1.0
            try:
                from astropy.io import fits as afits
                with afits.open(frame_path) as hdul:
                    exptime = float(hdul[0].header.get("EXPTIME", 1.0))
            except Exception:
                pass
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
            (mag_4, merr_4, ci_values), (passes_ci, passes_merr, keep) = apply_ci_cut(
                mag_txt, ci_threshold, merr_cut,
                filter_is_vband=("555" in filt or "F555" in filt),
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
            logger.warning("Photometry failed for filter %s: %s", filt, e)

    if not phot_rows:
        logger.warning("No photometry rows; skipping catalogue")
        return
    phot_df = pd.DataFrame(phot_rows)
    phot_path = config.catalogue_dir(galaxy_id) / (
        f"photometry_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    )
    phot_df.to_parquet(phot_path, index=False)
    logger.info("Wrote photometry parquet to %s", phot_path)
    cat_df = apply_catalogue_filters(
        phot_path, merr_cut=merr_cut, vband_filter=vband_name, required_filters=filters
    )
    cat_path = config.catalogue_dir(galaxy_id) / (
        f"catalogue_frame{frame_id}_{outname}_reff{reff:.2f}.parquet"
    )
    write_catalogue_parquet(cat_df, cat_path)
    logger.info("Wrote catalogue parquet to %s", cat_path)
