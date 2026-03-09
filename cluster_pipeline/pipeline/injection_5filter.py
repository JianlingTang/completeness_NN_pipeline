"""
Prepare matched clusters for 5-filter injection onto HLSP science images.

Photometry and CI cut must run on frames where the same matched clusters
(same coordinates on white and all 5 filters) are injected onto the real
HLSP ... F*W drc/sci FITS, not onto white synthetic images.

This module writes per-filter coord files (x y mag) used by
scripts/inject_clusters_to_5filters.py with --use_white so that BAOlab
injects those sources onto the HLSP science frame and writes
galaxy/filter/synthetic_fits/*.fits (science + injected) for photometry.
"""
from pathlib import Path

import numpy as np

from ..matching.coordinate_matcher import load_coords


def write_matched_coords_per_filter(
    matched_coords_path: Path,
    cluster_ids_path: Path,
    mag_vega_path: Path,
    white_dir: Path,
    frame_id: int,
    reff: float,
    outname: str,
    filter_names: list[str],
) -> None:
    """
    Write one coord file per filter: (x, y, mag) with same (x,y) for all filters;
    mag = mag_vega[cluster_id, ifilt] for each matched cluster.

    Files are written as white_dir / "{filt}_position_{frame_id}_{outname}_reff{reff:.2f}.txt"
    so that inject_clusters_to_5filters.py with --use_white can find them.

    Parameters
    ----------
    matched_coords_path : Path
        File with matched positions (x y) or (x y mag); only x,y are used.
    cluster_ids_path : Path
        One cluster_id per line, same order as matched_coords rows.
    mag_vega_path : Path
        .npy array shape (ncl, 5) with VEGA mag per filter (columns = filter_names order).
    white_dir : Path
        Directory to write coord files (e.g. galaxy_id/white).
    frame_id : int
        Frame index.
    reff : float
        Aperture reff (e.g. 3.0).
    outname : str
        Run name (e.g. test).
    filter_names : list of str
        Filter names in same order as mag_vega columns (e.g. F275W, F336W, F435W, F555W, F814W).
    """
    coords = load_coords(matched_coords_path)
    if len(coords) == 0:
        return
    if not cluster_ids_path.exists():
        raise FileNotFoundError(
            f"cluster_ids file required for injection: {cluster_ids_path}"
        )
    cids = [
        int(line.strip())
        for line in cluster_ids_path.read_text().splitlines()
        if line.strip()
    ]
    if len(cids) != len(coords):
        raise ValueError(
            f"cluster_ids length {len(cids)} != matched coords length {len(coords)}"
        )
    mag_vega = np.load(mag_vega_path)
    if mag_vega.ndim == 1:
        mag_vega = mag_vega.reshape(-1, 1)
    n_filters = min(len(filter_names), mag_vega.shape[1])
    white_dir = Path(white_dir)
    white_dir.mkdir(parents=True, exist_ok=True)
    reff_str = f"{reff:.2f}"
    for i in range(n_filters):
        filt = filter_names[i]
        out_path = white_dir / f"{filt}_position_{frame_id}_{outname}_reff{reff_str}.txt"
        lines = []
        for k, (x, y) in enumerate(coords):
            cid = cids[k]
            if cid < 0 or cid >= len(mag_vega):
                mag = np.nan
            else:
                mag = mag_vega[cid, i]
            lines.append(f"{x:.4f} {y:.4f} {mag:.4f}\n")
        out_path.write_text("".join(lines))
    return None
