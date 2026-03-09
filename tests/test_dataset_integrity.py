"""
Tests for cluster_pipeline.dataset.dataset_builder.
Focus: cluster_id alignment, dataset integrity, parquet/npy output.
"""

import numpy as np
import pandas as pd

from cluster_pipeline.data.schemas import get_required_columns
from cluster_pipeline.dataset.dataset_builder import build_dataset, write_dataset_npy, write_dataset_parquet


class TestBuildDataset:
    """build_dataset joins by cluster_id; detection_label aligned with in_catalogue."""

    def test_cluster_id_alignment(
        self,
        sample_injected_parquet,
        sample_match_parquet,
        sample_catalogue_parquet,
    ):
        df = build_dataset(
            sample_injected_parquet,
            sample_match_parquet,
            sample_catalogue_parquet,
        )
        assert "cluster_id" in df.columns
        assert "detection_label" in df.columns
        assert list(df["cluster_id"]) == [1, 2, 3]
        # detection_label = in_catalogue from catalogue (1, 0, 0)
        np.testing.assert_array_equal(df["detection_label"].values, np.array([1, 0, 0]))

    def test_dataset_row_has_required_columns(
        self,
        sample_injected_parquet,
        sample_match_parquet,
        sample_catalogue_parquet,
    ):
        df = build_dataset(
            sample_injected_parquet,
            sample_match_parquet,
            sample_catalogue_parquet,
        )
        required = get_required_columns("dataset_row")
        for c in required:
            assert c in df.columns, f"Missing column {c}"

    def test_mass_age_av_preserved_from_injected(
        self,
        sample_injected_parquet,
        sample_match_parquet,
        sample_catalogue_parquet,
    ):
        df = build_dataset(
            sample_injected_parquet,
            sample_match_parquet,
            sample_catalogue_parquet,
        )
        np.testing.assert_array_almost_equal(df["mass"].values, [1e4, 2e4, 3e4])
        np.testing.assert_array_almost_equal(df["age"].values, [1e8, 2e8, 3e8])
        np.testing.assert_array_almost_equal(df["av"].values, [0.1, 0.2, 0.3])


class TestWriteDatasetNpy:
    def test_writes_three_files(self, tmp_path_dir, sample_injected_parquet, sample_match_parquet, sample_catalogue_parquet):
        df = build_dataset(sample_injected_parquet, sample_match_parquet, sample_catalogue_parquet)
        write_dataset_npy(df, tmp_path_dir, prefix="dataset")
        assert (tmp_path_dir / "dataset_cluster_properties.npy").exists()
        assert (tmp_path_dir / "dataset_magnitudes.npy").exists()
        assert (tmp_path_dir / "dataset_detection_labels.npy").exists()

    def test_detection_labels_shape_matches_rows(self, tmp_path_dir, sample_injected_parquet, sample_match_parquet, sample_catalogue_parquet):
        df = build_dataset(sample_injected_parquet, sample_match_parquet, sample_catalogue_parquet)
        write_dataset_npy(df, tmp_path_dir, prefix="d")
        labels = np.load(tmp_path_dir / "d_detection_labels.npy")
        assert labels.shape[0] == len(df)
        np.testing.assert_array_equal(labels, df["detection_label"].values)


class TestWriteDatasetParquet:
    def test_roundtrip(self, tmp_path_dir, sample_injected_parquet, sample_match_parquet, sample_catalogue_parquet):
        df = build_dataset(sample_injected_parquet, sample_match_parquet, sample_catalogue_parquet)
        path = tmp_path_dir / "dataset.parquet"
        write_dataset_parquet(df, path)
        assert path.exists()
        df2 = pd.read_parquet(path)
        assert len(df2) == len(df)
        assert list(df2["cluster_id"]) == list(df["cluster_id"])
