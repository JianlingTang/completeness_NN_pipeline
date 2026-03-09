"""
Dataset builder: aggregate cluster properties, magnitudes, and detection labels into ML-ready arrays.
Outputs parquet tables and legacy .npy (cluster_properties, magnitudes, detection_labels).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schemas import get_required_columns


def build_dataset(
    injected_parquet_path: Path,
    match_parquet_path: Path,
    catalogue_parquet_path: Path,
    photometry_parquet_path: Path | None = None,
) -> pd.DataFrame:
    """
    Join injected clusters, match results, catalogue in_catalogue, and (optionally) photometry
    to produce one row per cluster with (M, T, Av, magnitudes) and detection_label C.

    Parameters
    ----------
    injected_parquet_path : Path
        Parquet from injection stage (cluster_id, mass, age, av, x, y, mag_f0, ...).
    match_parquet_path : Path
        Parquet from matching stage (cluster_id, matched, ...).
    catalogue_parquet_path : Path
        Parquet from catalogue stage (cluster_id, in_catalogue).
    photometry_parquet_path : Path, optional
        If provided, magnitudes are taken from photometry table (pivot by filter); else from injected.

    Returns
    -------
    pd.DataFrame
        Columns per DATASET_ROW_SCHEMA (cluster_id, galaxy_id, frame_id, reff, mass, age, av, mag_f0..4, detection_label).
    """
    inj = pd.read_parquet(injected_parquet_path)
    match_df = pd.read_parquet(match_parquet_path)
    cat = pd.read_parquet(catalogue_parquet_path)
    # Detection label = in_catalogue (after CI + quality cuts)
    match_df = match_df.merge(
        cat[["cluster_id", "galaxy_id", "frame_id", "reff", "in_catalogue"]],
        on=["cluster_id", "galaxy_id", "frame_id", "reff"],
        how="left",
    )
    match_df["detection_label"] = match_df["in_catalogue"].fillna(0).astype(np.int8)
    # Join with injected for mass, age, av, magnitudes
    merge_cols = ["cluster_id", "galaxy_id", "frame_id", "reff"]
    base = list(set(merge_cols) & set(inj.columns))
    df = match_df.merge(inj[[c for c in base + ["mass", "age", "av"] + [c for c in inj.columns if c.startswith("mag_f")] if c in inj.columns]], on=base, how="left")
    # Ensure mag_f0..mag_f4 exist
    for i in range(5):
        if f"mag_f{i}" not in df.columns:
            df[f"mag_f{i}"] = np.nan
    return df[get_required_columns("dataset_row")]


def write_dataset_npy(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "dataset",
) -> None:
    """
    Write cluster_properties.npy, magnitudes.npy, detection_labels.npy (legacy format).
    Aligned by row index (cluster_id order preserved).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_cluster_properties.npy", df[["mass", "age", "av"]].values)
    mag_cols = [f"mag_f{i}" for i in range(5) if f"mag_f{i}" in df.columns]
    if mag_cols:
        np.save(output_dir / f"{prefix}_magnitudes.npy", df[mag_cols].values)
    else:
        np.save(output_dir / f"{prefix}_magnitudes.npy", np.zeros((len(df), 5)) * np.nan)
    np.save(output_dir / f"{prefix}_detection_labels.npy", df["detection_label"].values)


def write_dataset_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write full dataset table to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
