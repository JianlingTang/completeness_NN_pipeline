"""
Tests for cluster_pipeline.catalogue.label_builder.
Focus: cluster_id alignment, detection label consistency.
"""

import numpy as np
import pandas as pd

from cluster_pipeline.catalogue.label_builder import build_final_detection, save_final_detection


class TestBuildFinalDetection:
    """build_final_detection: labels aligned with cluster_id order from catalogue or match results."""

    def test_from_catalogue_only(self, tmp_path_dir):
        cat = pd.DataFrame({
            "cluster_id": [3, 1, 2],
            "galaxy_id": ["G"] * 3,
            "frame_id": [0] * 3,
            "reff": [5.0] * 3,
            "in_catalogue": np.int8([0, 1, 1]),
        })
        path = tmp_path_dir / "cat.parquet"
        cat.to_parquet(path, index=False)
        labels = build_final_detection(path, match_results_parquet_path=None)
        # Order from catalogue unique: [3, 1, 2]
        assert labels.dtype == np.uint8
        assert len(labels) == 3
        cid_to_label = {3: 0, 1: 1, 2: 1}
        np.testing.assert_array_equal(
            labels,
            [cid_to_label[3], cid_to_label[1], cid_to_label[2]],
        )

    def test_aligned_to_match_results_order(self, tmp_path_dir):
        cat = pd.DataFrame({
            "cluster_id": [1, 2, 3],
            "galaxy_id": ["G"] * 3,
            "frame_id": [0] * 3,
            "reff": [5.0] * 3,
            "in_catalogue": np.int8([1, 0, 1]),
        })
        match_df = pd.DataFrame({
            "cluster_id": [3, 1, 2],
            "galaxy_id": ["G"] * 3,
        })
        cat_path = tmp_path_dir / "cat.parquet"
        match_path = tmp_path_dir / "match.parquet"
        cat.to_parquet(cat_path, index=False)
        match_df.to_parquet(match_path, index=False)
        labels = build_final_detection(cat_path, match_results_parquet_path=match_path)
        # Order from match: [3, 1, 2] -> labels [1, 1, 0]
        np.testing.assert_array_equal(labels, np.array([1, 1, 0], dtype=np.uint8))

    def test_missing_cluster_id_gets_zero(self, tmp_path_dir):
        cat = pd.DataFrame({
            "cluster_id": [1, 2],
            "galaxy_id": ["G"] * 2,
            "frame_id": [0] * 2,
            "reff": [5.0] * 2,
            "in_catalogue": np.int8([1, 0]),
        })
        match_df = pd.DataFrame({"cluster_id": [1, 99, 2]})
        cat_path = tmp_path_dir / "cat.parquet"
        match_path = tmp_path_dir / "match.parquet"
        cat.to_parquet(cat_path, index=False)
        match_df.to_parquet(match_path, index=False)
        labels = build_final_detection(cat_path, match_results_parquet_path=match_path)
        assert labels[0] == 1
        assert labels[1] == 0
        assert labels[2] == 0


class TestSaveFinalDetection:
    """save_final_detection: .npy roundtrip."""

    def test_roundtrip(self, tmp_path_dir):
        labels = np.array([1, 0, 1], dtype=np.uint8)
        path = tmp_path_dir / "sub" / "final_detection.npy"
        save_final_detection(labels, path)
        assert path.exists()
        back = np.load(path)
        np.testing.assert_array_equal(back, labels)
