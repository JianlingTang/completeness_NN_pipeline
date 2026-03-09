"""
Tests for cluster_pipeline.dataset.dataset_builder.
Focus: data alignment, cluster_id integrity, dataset consistency (join order, columns).
"""

import numpy as np
import pandas as pd

from cluster_pipeline.data.schemas import get_required_columns
from cluster_pipeline.dataset.dataset_builder import (
    build_dataset,
    write_dataset_npy,
    write_dataset_parquet,
)


class TestBuildDataset:
    """build_dataset: join injected + match + catalogue -> one row per cluster, detection_label from in_catalogue."""

    def test_output_has_required_columns(
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
        for col in required:
            assert col in df.columns, f"Missing column {col}"
        assert len(df) == 3

    def test_detection_label_from_catalogue(
        self,
        sample_injected_parquet,
        sample_match_parquet,
        sample_catalogue_parquet,
    ):
        # catalogue has in_catalogue [1,0,0] for cluster_id 1,2,3
        df = build_dataset(
            sample_injected_parquet,
            sample_match_parquet,
            sample_catalogue_parquet,
        )
        assert list(df["detection_label"].values) == [1, 0, 0]
        assert set(df["detection_label"].unique()).issubset({0, 1})

    def test_cluster_id_integrity_preserved(
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
        assert list(df["cluster_id"]) == [1, 2, 3]
        assert list(df["galaxy_id"]) == ["NGC123"] * 3
        assert list(df["reff"]) == [5.0, 5.0, 5.0]

    def test_mass_age_av_from_injected(
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

    def test_mag_columns_present(
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
        for i in range(5):
            assert f"mag_f{i}" in df.columns
        np.testing.assert_array_almost_equal(df["mag_f0"].values, [18.0, 18.5, 19.0])


class TestWriteDatasetNpy:
    """write_dataset_npy: row alignment (cluster_id order preserved in .npy files)."""

    def test_writes_three_files(self, tmp_path_dir):
        df = pd.DataFrame({
            "cluster_id": [1, 2, 3],
            "galaxy_id": ["G"] * 3,
            "frame_id": [0] * 3,
            "reff": [5.0] * 3,
            "mass": [1.0, 2.0, 3.0],
            "age": [1e8] * 3,
            "av": [0.1] * 3,
            "mag_f0": [18.0, 18.5, 19.0],
            "mag_f1": [18.1, 18.6, 19.1],
            "mag_f2": [18.2, 18.7, 19.2],
            "mag_f3": [18.3, 18.8, 19.3],
            "mag_f4": [18.4, 18.9, 19.4],
            "detection_label": np.int8([1, 0, 1]),
        })
        write_dataset_npy(df, tmp_path_dir, prefix="dataset")
        assert (tmp_path_dir / "dataset_cluster_properties.npy").exists()
        assert (tmp_path_dir / "dataset_magnitudes.npy").exists()
        assert (tmp_path_dir / "dataset_detection_labels.npy").exists()

    def test_npy_alignment_by_row_index(self, tmp_path_dir):
        df = pd.DataFrame({
            "cluster_id": [10, 20, 30],
            "galaxy_id": ["G"] * 3,
            "frame_id": [0] * 3,
            "reff": [5.0] * 3,
            "mass": [1.0, 2.0, 3.0],
            "age": [1e8] * 3,
            "av": [0.1] * 3,
            "mag_f0": [17.0, 18.0, 19.0],
            "mag_f1": [17.1, 18.1, 19.1],
            "mag_f2": [17.2, 18.2, 19.2],
            "mag_f3": [17.3, 18.3, 19.3],
            "mag_f4": [17.4, 18.4, 19.4],
            "detection_label": np.int8([1, 0, 1]),
        })
        write_dataset_npy(df, tmp_path_dir, prefix="ds")
        props = np.load(tmp_path_dir / "ds_cluster_properties.npy")
        mags = np.load(tmp_path_dir / "ds_magnitudes.npy")
        labels = np.load(tmp_path_dir / "ds_detection_labels.npy")
        assert props.shape[0] == 3
        assert mags.shape[0] == 3
        assert labels.shape[0] == 3
        np.testing.assert_array_almost_equal(props[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(mags[:, 0], [17.0, 18.0, 19.0])
        np.testing.assert_array_equal(labels, [1, 0, 1])


class TestWriteDatasetParquet:
    """write_dataset_parquet: full table to parquet."""

    def test_roundtrip(self, tmp_path_dir):
        df = pd.DataFrame({
            "cluster_id": [1],
            "galaxy_id": ["G"],
            "frame_id": [0],
            "reff": [5.0],
            "mass": [1e4],
            "age": [1e8],
            "av": [0.2],
            "mag_f0": [18.0],
            "mag_f1": [18.1],
            "mag_f2": [18.2],
            "mag_f3": [18.3],
            "mag_f4": [18.4],
            "detection_label": np.int8([1]),
        })
        path = tmp_path_dir / "out.parquet"
        write_dataset_parquet(df, path)
        assert path.exists()
        back = pd.read_parquet(path)
        assert back.shape[0] == 1
        assert back["cluster_id"].iloc[0] == 1
