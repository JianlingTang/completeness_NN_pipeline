"""
Coordinate matcher: match injected (synthetic) positions to detected (SExtractor) positions.
Uses KD-tree for nearest-neighbor matching within a pixel tolerance.
No global state; inputs/outputs are paths or arrays.

Coord format (aligned with scripts/legus_original_pipeline.py and BAOlab):
  All position files: (x y [mag]), x = column = NAXIS1, y = row = NAXIS2 (FITS/IRAF).
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..data.models import MatchResult


def load_coords(path: Path) -> np.ndarray:
    """
    Load coordinate file: either two-column (x y) or three-column (x y mag).
    Returns (N, 2) array of (x, y).
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :2].astype(np.float64)


def load_coords_white_position(path: Path) -> np.ndarray:
    """
    Load white_position file (same format as legus_original_pipeline.py coord files).
    File columns: (x y mag), x = column = NAXIS1, y = row = NAXIS2.
    Returns (N, 2) array of (x, y) for SExtractor/matcher.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :2].astype(np.float64)


def match_coordinates(
    injected_coords: Path | np.ndarray,
    detected_coords: Path | np.ndarray,
    tolerance_pix: float = 3.0,
    cluster_ids: list[int] | None = None,
) -> MatchResult:
    """
    Match injected coordinates to detected coordinates using KD-tree.
    Each injected point is matched to the nearest detected point if within tolerance_pix.

    Parameters
    ----------
    injected_coords : Path or (N, 2) array
        Injected (synthetic) positions (x, y).
    detected_coords : Path or (M, 2) array
        Detected positions from SExtractor (x, y).
    tolerance_pix : float
        Maximum distance (pixels) to consider a match.
    cluster_ids : list of int, optional
        Stable cluster_id per injected row (same order). If None, uses range(n_injected).

    Returns
    -------
    MatchResult
        matched_indices, matched_positions, cluster_ids, detection_labels, etc.
    """
    if isinstance(injected_coords, Path):
        injected = load_coords(injected_coords)
        injected_path = injected_coords
    else:
        injected = np.asarray(injected_coords, dtype=np.float64)
        if injected.ndim == 1:
            injected = injected.reshape(1, -1)
        injected = injected[:, :2]
        injected_path = Path(".")
    if isinstance(detected_coords, Path):
        detected = load_coords(detected_coords)
        detected_path = detected_coords
    else:
        detected = np.asarray(detected_coords, dtype=np.float64)
        if detected.ndim == 1:
            detected = detected.reshape(1, -1)
        detected = detected[:, :2]
        detected_path = Path(".")

    n_injected = len(injected)
    if cluster_ids is None:
        cluster_ids = list(range(n_injected))
    if len(cluster_ids) != n_injected:
        raise ValueError("cluster_ids length must match number of injected coordinates")

    if n_injected == 0:
        return MatchResult(
            injected_path=injected_path,
            detected_path=detected_path,
            cluster_ids=cluster_ids,
            matched_indices=[],
            matched_positions=[],
            n_injected=0,
            n_matched=0,
            tolerance_pix=tolerance_pix,
        )
    if len(detected) == 0:
        return MatchResult(
            injected_path=injected_path,
            detected_path=detected_path,
            cluster_ids=cluster_ids,
            matched_indices=[],
            matched_positions=[],
            n_injected=n_injected,
            n_matched=0,
            tolerance_pix=tolerance_pix,
        )

    tree = cKDTree(detected)
    distances, indices = tree.query(injected, k=1, distance_upper_bound=tolerance_pix)
    matched_mask = np.isfinite(distances) & (distances < tolerance_pix)
    matched_indices = np.where(matched_mask)[0].tolist()
    matched_positions = detected[indices[matched_mask]].tolist()

    return MatchResult(
        injected_path=injected_path,
        detected_path=detected_path,
        cluster_ids=cluster_ids,
        matched_indices=matched_indices,
        matched_positions=matched_positions,
        n_injected=n_injected,
        n_matched=len(matched_indices),
        tolerance_pix=tolerance_pix,
    )


class CoordinateMatcher:
    """
    Stateless coordinate matcher; wraps match_coordinates with config.
    Use when you want to pass a config object and reuse tolerance.
    """

    def __init__(self, tolerance_pix: float = 3.0):
        self.tolerance_pix = tolerance_pix

    def match(
        self,
        injected_coords: Path | np.ndarray,
        detected_coords: Path | np.ndarray,
        cluster_ids: list[int] | None = None,
    ) -> MatchResult:
        return match_coordinates(
            injected_coords,
            detected_coords,
            tolerance_pix=self.tolerance_pix,
            cluster_ids=cluster_ids,
        )

    def write_matched_coords(
        self,
        match_result: MatchResult,
        out_path: Path,
        detected_coords_path: Path,
        include_mag: bool = False,
        include_cluster_id: bool = False,
    ) -> None:
        """
        Write matched coordinates to a file in IRAF/BAOlab format (x y [mag]).
        Uses detected positions and optionally adds magnitude from detected catalog.
        When include_cluster_id is True, also writes a sidecar file with one cluster_id per line
        (same order as rows) for photometry stage. IRAF uses only x y [mag]; do not add 4th col to coords.
        """
        if match_result.n_matched == 0:
            out_path.write_text("")
            if include_cluster_id:
                (out_path.parent / (out_path.stem + "_cluster_ids.txt")).write_text("")
            return
        detected = load_coords(detected_coords_path)
        if include_mag and detected.shape[1] >= 3:
            mags = detected[:, 2]
        else:
            mags = np.zeros(match_result.n_matched)
        lines = []
        for i, (x, y) in enumerate(match_result.matched_positions):
            m = mags[i] if i < len(mags) else 0.0
            lines.append(f"{x:.2f} {y:.2f} {m:.4f}\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines))
        if include_cluster_id:
            matched_cids = match_result.get_matched_cluster_ids()
            cid_path = out_path.parent / (out_path.stem + "_cluster_ids.txt")
            cid_path.write_text("\n".join(str(c) for c in matched_cids) + "\n")
