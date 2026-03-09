"""
Tests for cluster_pipeline.catalogue.catalogue_filters.
Focus: in_catalogue logic, schema consistency, cluster_id integrity.
"""

import numpy as np
import pandas as pd

from cluster_pipeline.catalogue.catalogue_filters import (
    apply_catalogue_filters,
    write_catalogue_parquet,
)
from cluster_pipeline.data.schemas import CATALOGUE_FILTERS_SCHEMA


class TestApplyCatalogueFilters:
    """apply_catalogue_filters: one row per (cluster_id, galaxy_id, frame_id, reff), in_catalogue = passes_ci & passes_merr & passes_multiband."""

    def test_output_schema(self, sample_photometry_parquet):
        df = apply_catalogue_filters(
            sample_photometry_parquet,
            merr_cut=0.3,
            required_filters=["F275W", "F336W", "F438W", "F555W", "F814W"],
        )
        for col in CATALOGUE_FILTERS_SCHEMA:
            assert col in df.columns
        assert len(df) == 3

    def test_cluster_id_integrity(self, sample_photometry_parquet):
        df = apply_catalogue_filters(
            sample_photometry_parquet,
            merr_cut=0.5,
            required_filters=["F275W", "F336W", "F438W", "F555W", "F814W"],
        )
        assert set(df["cluster_id"]) == {1, 2, 3}
        assert df["galaxy_id"].nunique() == 1

    def test_passes_merr_all_under_cut(self, tmp_path_dir):
        # All merr 0.1 < 0.3 -> passes_merr=1
        rows = []
        for cid in [1]:
            for fname in ["F555W", "F814W"]:
                rows.append({
                    "cluster_id": cid,
                    "galaxy_id": "G",
                    "frame_id": 0,
                    "reff": 5.0,
                    "filter_name": fname,
                    "mag": 18.0,
                    "merr": 0.1,
                    "mag_1px": 18.0,
                    "mag_3px": 18.5,
                    "ci": 0.5,
                })
        df_in = pd.DataFrame(rows)
        path = tmp_path_dir / "phot.parquet"
        df_in.to_parquet(path, index=False)
        out = apply_catalogue_filters(path, merr_cut=0.3, required_filters=None)
        assert out["passes_merr"].iloc[0] == 1

    def test_in_catalogue_binary(self, sample_photometry_parquet):
        df = apply_catalogue_filters(
            sample_photometry_parquet,
            merr_cut=0.5,
            required_filters=["F275W", "F336W", "F438W", "F555W", "F814W"],
        )
        assert set(df["in_catalogue"].unique()).issubset({0, 1})


class TestWriteCatalogueParquet:
    """write_catalogue_parquet: writes parquet with correct columns."""

    def test_creates_file(self, tmp_path_dir):
        df = pd.DataFrame({
            "cluster_id": [1],
            "galaxy_id": ["G"],
            "frame_id": [0],
            "reff": [5.0],
            "passes_ci": np.int8([1]),
            "passes_merr": np.int8([1]),
            "passes_multiband": np.int8([1]),
            "in_catalogue": np.int8([1]),
        })
        path = tmp_path_dir / "cat.parquet"
        write_catalogue_parquet(df, path)
        assert path.exists()
        back = pd.read_parquet(path)
        assert back["cluster_id"].iloc[0] == 1
