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
    """apply_catalogue_filters: LEGUS Stage A (CI + V and B|I merr) & Stage B (>=4 bands merr + M_V)."""

    def test_output_schema(self, sample_photometry_parquet):
        df = apply_catalogue_filters(
            sample_photometry_parquet,
            merr_cut=0.3,
        )
        for col in CATALOGUE_FILTERS_SCHEMA:
            assert col in df.columns
        assert len(df) == 3

    def test_cluster_id_integrity(self, sample_photometry_parquet):
        df = apply_catalogue_filters(
            sample_photometry_parquet,
            merr_cut=0.5,
        )
        assert set(df["cluster_id"]) == {1, 2, 3}
        assert df["galaxy_id"].nunique() == 1

    def test_passes_stage1_stage2_merr(self, tmp_path_dir):
        # V, B, I merr 0.1 -> Stage A pass; 5 filters all 0.1 -> Stage B pass (>=4 good)
        rows = []
        for cid in [1]:
            for fname in ["F275W", "F336W", "F435W", "F555W", "F814W"]:
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
        out = apply_catalogue_filters(path, merr_cut=0.3)
        assert out["passes_stage1_merr"].iloc[0] == 1
        assert out["passes_stage2_merr"].iloc[0] == 1

    def test_in_catalogue_binary(self, sample_photometry_parquet):
        df = apply_catalogue_filters(
            sample_photometry_parquet,
            merr_cut=0.5,
        )
        assert set(df["in_catalogue"].unique()).issubset({0, 1})

    def test_mv_cut_turnover_at_faint_end_not_bright(self, tmp_path):
        """M_V <= -6: keep bright (low m_V), discard faint (high m_V). Turnover at m_V ~ dmod - 6 (~24 mag), not at 20."""
        dmod = 29.98
        rows = []
        for cid, v_mag in [(1, 20.0), (2, 25.0)]:
            for fname in ["F275W", "F336W", "F435W", "F555W", "F814W"]:
                mag = v_mag if fname == "F555W" else 21.0
                rows.append({
                    "cluster_id": cid,
                    "galaxy_id": "G",
                    "frame_id": 0,
                    "reff": 5.0,
                    "filter_name": fname,
                    "mag": mag,
                    "merr": 0.1,
                    "mag_1px": mag,
                    "mag_3px": mag + 0.3,
                    "ci": 0.3,
                })
        path = tmp_path / "phot.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        df = apply_catalogue_filters(path, merr_cut=0.3, dmod=dmod)
        # Bright (m_V=20) -> M_V = -9.98 <= -6 -> pass
        assert df[df["cluster_id"] == 1]["passes_MV"].iloc[0] == 1
        # Faint (m_V=25) -> M_V = -4.98 > -6 -> fail
        assert df[df["cluster_id"] == 2]["passes_MV"].iloc[0] == 0


class TestWriteCatalogueParquet:
    """write_catalogue_parquet: writes parquet with correct columns."""

    def test_creates_file(self, tmp_path_dir):
        df = pd.DataFrame({
            "cluster_id": [1],
            "galaxy_id": ["G"],
            "frame_id": [0],
            "reff": [5.0],
            "passes_ci": np.int8([1]),
            "passes_stage1_merr": np.int8([1]),
            "passes_stage2_merr": np.int8([1]),
            "passes_MV": np.int8([1]),
            "in_catalogue": np.int8([1]),
        })
        path = tmp_path_dir / "cat.parquet"
        write_catalogue_parquet(df, path)
        assert path.exists()
        back = pd.read_parquet(path)
        assert back["cluster_id"].iloc[0] == 1
