"""Shared pytest fixtures and path setup for cluster pipeline tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root (parent of tests/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


@pytest.fixture
def project_root():
    """Project root directory (parent of tests/)."""
    return ROOT


@pytest.fixture
def tmp_path_dir(tmp_path):
    """Temporary directory for tests that expect a tmp_path_dir fixture (alias for tmp_path)."""
    return tmp_path


# ---- Coordinate matcher fixtures (for tests/test_coordinate_matcher.py) ----
@pytest.fixture
def sample_injected_coords(tmp_path):
    """Path to a two-column (x y) coords file with 3 rows: (10,20), (30,40), (50,60)."""
    p = tmp_path / "injected.coo"
    p.write_text("10 20\n30 40\n50 60\n")
    return p


@pytest.fixture
def sample_detected_coords(tmp_path):
    """Path to detected coords: (11,21), (31,41), (100,200); first two within 3 px of injected."""
    p = tmp_path / "detected.coo"
    p.write_text("11 21\n31 41\n100 200\n")
    return p


@pytest.fixture
def sample_coords_three_col(tmp_path):
    """Path to a three-column (x y mag) coords file with 2 rows."""
    p = tmp_path / "three_col.coo"
    p.write_text("1 2 18.0\n4 5 19.0\n")
    return p


@pytest.fixture
def sample_cluster_ids():
    """Cluster IDs for 3 injected clusters (used with sample_injected_coords)."""
    return [101, 102, 103]


# ---- Catalogue filters fixture (for tests/test_catalogue_filters.py) ----
@pytest.fixture
def sample_photometry_parquet(tmp_path):
    """Path to photometry parquet with 3 clusters, multiple filters; used by apply_catalogue_filters."""
    rows = []
    for cid in [1, 2, 3]:
        for fname in ["F275W", "F336W", "F438W", "F555W", "F814W"]:
            rows.append({
                "cluster_id": cid,
                "galaxy_id": "G",
                "frame_id": 0,
                "reff": 5.0,
                "filter_name": fname,
                "mag": 18.0 + cid * 0.1,
                "merr": 0.2,
                "mag_1px": 18.0,
                "mag_3px": 18.5,
                "ci": 0.5,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "photometry.parquet"
    df.to_parquet(path, index=False)
    return path


# ---- Dataset builder fixtures (for tests/test_dataset_builder.py, test_dataset_integrity.py) ----
@pytest.fixture
def sample_injected_parquet(tmp_path):
    """Path to injected clusters parquet: 3 clusters with mass, age, av, mag_f0..4."""
    df = pd.DataFrame({
        "cluster_id": [1, 2, 3],
        "galaxy_id": ["NGC123"] * 3,
        "frame_id": [0] * 3,
        "reff": [5.0] * 3,
        "mass": [1e4, 2e4, 3e4],
        "age": [1e8, 2e8, 3e8],
        "av": [0.1, 0.2, 0.3],
        "radius": [1.0] * 3,
        "x": [10.0, 20.0, 30.0],
        "y": [10.0, 20.0, 30.0],
        "mag_f0": [18.0, 18.5, 19.0],
        "mag_f1": [18.1, 18.6, 19.1],
        "mag_f2": [18.2, 18.7, 19.2],
        "mag_f3": [18.3, 18.8, 19.3],
        "mag_f4": [18.4, 18.9, 19.4],
    })
    path = tmp_path / "injected.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def sample_match_parquet(tmp_path):
    """Path to match results parquet: 3 clusters, galaxy_id/frame_id/reff for join."""
    df = pd.DataFrame({
        "cluster_id": [1, 2, 3],
        "galaxy_id": ["NGC123"] * 3,
        "frame_id": [0] * 3,
        "reff": [5.0] * 3,
        "matched": np.int8([1, 1, 1]),
        "injected_x": [10.0, 20.0, 30.0],
        "injected_y": [10.0, 20.0, 30.0],
        "detected_x": [10.1, 20.1, 30.1],
        "detected_y": [10.1, 20.1, 30.1],
    })
    path = tmp_path / "match.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def sample_catalogue_parquet(tmp_path):
    """Path to catalogue parquet: in_catalogue [1, 0, 0] for cluster_id 1,2,3."""
    df = pd.DataFrame({
        "cluster_id": [1, 2, 3],
        "galaxy_id": ["NGC123"] * 3,
        "frame_id": [0] * 3,
        "reff": [5.0] * 3,
        "passes_ci": np.int8([1, 1, 1]),
        "passes_stage1_merr": np.int8([1, 1, 1]),
        "passes_stage2_merr": np.int8([1, 1, 1]),
        "passes_MV": np.int8([1, 1, 1]),
        "in_catalogue": np.int8([1, 0, 0]),
    })
    path = tmp_path / "catalogue.parquet"
    df.to_parquet(path, index=False)
    return path

