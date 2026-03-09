"""
Diagnostics after stage 3: completeness vs magnitude.
Pipeline writes (mag, matched) per frame/reff when running stage 3; this module
aggregates and plots completeness = fraction detected per magnitude bin.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import PipelineConfig
from ..data.models import MatchResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def load_coords_with_mag(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load coordinate file: two-column (x y) or three-column (e.g. y x mag).
    Returns (coords (N,2), mags (N,) or None).
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    coords = data[:, :2].astype(np.float64)
    mags = data[:, 2] if data.shape[1] >= 3 else None
    return coords, mags


def write_match_summary(
    coord_path: Path,
    match_result: MatchResult,
    out_path: Path,
) -> None:
    """
    Write (mag, matched) per injected source for later completeness diagnostics.
    coord_path: injected coords file (2 or 3 columns; if 3, last is mag).
    """
    coords, mags = load_coords_with_mag(coord_path)
    n = len(coords)
    matched = np.zeros(n, dtype=np.int8)
    for i in match_result.matched_indices:
        if 0 <= i < n:
            matched[i] = 1
    if mags is None:
        mags = np.full(n, np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_path,
        np.column_stack([mags, matched]),
        fmt="%.4f %d",
        header="mag matched",
        comments="",
    )


def load_match_summaries(
    diagnostics_dir: Path,
    outname: str = "pipeline",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all match_summary_frame*_reff*_{outname}.txt in diagnostics_dir.
    Returns (mags, matched) concatenated over all frames/reff.
    """
    pattern = f"match_summary_frame*_reff*_{outname}.txt"
    files = sorted(diagnostics_dir.glob(pattern))
    if not files:
        return np.array([]), np.array([])
    mags_list, matched_list = [], []
    for p in files:
        data = np.loadtxt(p)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        mags_list.append(data[:, 0])
        matched_list.append(data[:, 1].astype(np.int8))
    return np.concatenate(mags_list), np.concatenate(matched_list)


def completeness_per_bin(
    mags: np.ndarray,
    matched: np.ndarray,
    mag_bins: np.ndarray | None = None,
    mag_min: float | None = None,
    mag_max: float | None = None,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin by magnitude and compute completeness = matched / total per bin.
    Returns (bin_centers, completeness, bin_edges).
    """
    if mag_bins is None:
        if mag_min is None:
            mag_min = np.nanmin(mags) if len(mags) else 0
        if mag_max is None:
            mag_max = np.nanmax(mags) if len(mags) else 1
        mag_bins = np.linspace(mag_min, mag_max, n_bins + 1)
    bin_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    total = np.histogram(mags, bins=mag_bins)[0]
    det = np.histogram(mags[matched == 1], bins=mag_bins)[0]
    comp = np.where(total > 0, det / total.astype(float), np.nan)
    return bin_centers, comp, mag_bins


def plot_completeness_diagnostics(
    galaxy_id: str,
    config: PipelineConfig,
    outname: str = "pipeline",
    mag_bins: np.ndarray | None = None,
    mag_min: float | None = None,
    mag_max: float | None = None,
    n_bins: int = 15,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """
    Plot completeness (y) vs magnitude (x) from match summaries written by stage 3.
    diagnostics_dir contains match_summary_*.txt files.

    Parameters
    ----------
    galaxy_id : str
        Galaxy ID (same as used for pipeline run).
    config : PipelineConfig
        Config with paths.
    outname : str
        Run name (same as pipeline outname).
    mag_bins, mag_min, mag_max, n_bins
        Binning for magnitude axis.
    ax : matplotlib.axes.Axes, optional
        If given, plot here; else create new figure.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    diagnostics_dir = config.diagnostics_dir(galaxy_id)
    mags, matched = load_match_summaries(diagnostics_dir, outname=outname)
    if len(mags) == 0:
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title(title or f"{galaxy_id} (no match summary data)")
        return ax

    bin_centers, comp, edges = completeness_per_bin(
        mags, matched,
        mag_bins=mag_bins,
        mag_min=mag_min,
        mag_max=mag_max,
        n_bins=n_bins,
    )
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(bin_centers, comp, "o-", label="completeness")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Completeness")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or f"{galaxy_id} completeness vs magnitude ({outname})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
