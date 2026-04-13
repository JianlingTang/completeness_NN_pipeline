"""Unit tests for cluster_pipeline.pipeline.pipeline_runner."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.data.models import MatchResult
from cluster_pipeline.pipeline.pipeline_runner import (
    _build_match_results_df,
    _write_white_match_labels,
)


def _make_match_result(
    n_injected: int = 3,
    matched_indices: list[int] | None = None,
    matched_positions: list[tuple[float, float]] | None = None,
    cluster_ids: list[int] | None = None,
) -> MatchResult:
    if matched_indices is None:
        matched_indices = [0, 2]
    if matched_positions is None:
        matched_positions = [(1.0, 2.0), (5.0, 6.0)]
    if cluster_ids is None:
        cluster_ids = list(range(n_injected))
    return MatchResult(
        injected_path=Path("."),
        detected_path=Path("."),
        cluster_ids=cluster_ids,
        matched_indices=matched_indices,
        matched_positions=matched_positions,
        n_injected=n_injected,
        n_matched=len(matched_indices),
        tolerance_pix=3.0,
    )


class TestBuildMatchResultsDf:
    """_build_match_results_df produces correct parquet-ready DataFrame."""

    def test_columns_present(self):
        match_result = _make_match_result(n_injected=2, matched_indices=[0])
        injected_xy = np.array([[10.0, 20.0], [30.0, 40.0]])
        df = _build_match_results_df(
            match_result, "gal1", frame_id=0, reff=3.0, injected_xy=injected_xy
        )
        expected_cols = {
            "cluster_id", "galaxy_id", "frame_id", "reff", "matched",
            "injected_x", "injected_y", "detected_x", "detected_y",
        }
        assert set(df.columns) == expected_cols

    def test_one_row_per_injected(self):
        n = 4
        match_result = _make_match_result(n_injected=n, matched_indices=[1, 3])
        injected_xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        df = _build_match_results_df(
            match_result, "g", frame_id=1, reff=5.0, injected_xy=injected_xy
        )
        assert len(df) == n

    def test_matched_flag_and_coords(self):
        match_result = _make_match_result(
            n_injected=3,
            matched_indices=[0, 2],
            matched_positions=[(10.5, 20.5), (50.5, 60.5)],
        )
        injected_xy = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        df = _build_match_results_df(
            match_result, "gal", frame_id=0, reff=3.0, injected_xy=injected_xy
        )
        assert df["matched"].tolist() == [1, 0, 1]
        assert df["galaxy_id"].iloc[0] == "gal"
        assert df["frame_id"].iloc[0] == 0
        assert df["reff"].iloc[0] == 3.0
        # Injected coords
        assert df["injected_x"].tolist() == [10.0, 30.0, 50.0]
        assert df["injected_y"].tolist() == [20.0, 40.0, 60.0]
        # Detected: first row matched -> (10.5, 20.5), second unmatched -> nan, third -> (50.5, 60.5)
        assert df["detected_x"].iloc[0] == pytest.approx(10.5)
        assert df["detected_y"].iloc[0] == pytest.approx(20.5)
        assert np.isnan(df["detected_x"].iloc[1]) and np.isnan(df["detected_y"].iloc[1])
        assert df["detected_x"].iloc[2] == pytest.approx(50.5)
        assert df["detected_y"].iloc[2] == pytest.approx(60.5)

    def test_cluster_ids_preserved(self):
        match_result = _make_match_result(
            n_injected=2, cluster_ids=[100, 200], matched_indices=[0]
        )
        injected_xy = np.array([[1.0, 1.0], [2.0, 2.0]])
        df = _build_match_results_df(
            match_result, "g", frame_id=0, reff=1.0, injected_xy=injected_xy
        )
        assert df["cluster_id"].tolist() == [100, 200]

    def test_empty_injected(self):
        match_result = _make_match_result(n_injected=0, matched_indices=[], matched_positions=[])
        injected_xy = np.array([]).reshape(0, 2)
        df = _build_match_results_df(
            match_result, "g", frame_id=0, reff=1.0, injected_xy=injected_xy
        )
        assert len(df) == 0
        # Empty list of rows yields empty DataFrame; columns present when there is at least one row
        assert isinstance(df, pd.DataFrame)


class TestWriteWhiteMatchLabels:
    """_write_white_match_labels writes correct numpy file."""

    def test_writes_file(self, tmp_path):
        match_result = _make_match_result(n_injected=3, matched_indices=[0, 2])
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_white_match_labels(
            match_result, labels_dir, frame_id=1, outname="test", reff=3.0
        )
        path = labels_dir / "detection_labels_white_match_frame1_test_reff3.00.npy"
        assert path.exists()
        arr = np.load(path)
        assert arr.dtype == np.uint8 or arr.dtype.kind == "u"
        assert arr.shape == (3,)
        assert list(arr) == [1, 0, 1]

    def test_filename_format(self, tmp_path):
        match_result = _make_match_result(n_injected=1, matched_indices=[0])
        _write_white_match_labels(
            match_result, tmp_path, frame_id=5, outname="run1", reff=10.0
        )
        path = tmp_path / "detection_labels_white_match_frame5_run1_reff10.00.npy"
        assert path.exists()

    def test_all_matched(self, tmp_path):
        match_result = _make_match_result(n_injected=2, matched_indices=[0, 1])
        _write_white_match_labels(
            match_result, tmp_path, frame_id=0, outname="x", reff=1.0
        )
        path = tmp_path / "detection_labels_white_match_frame0_x_reff1.00.npy"
        assert np.array_equal(np.load(path), np.array([1, 1], dtype=np.uint8))

    def test_none_matched(self, tmp_path):
        match_result = _make_match_result(n_injected=2, matched_indices=[])
        _write_white_match_labels(
            match_result, tmp_path, frame_id=0, outname="x", reff=1.0
        )
        path = tmp_path / "detection_labels_white_match_frame0_x_reff1.00.npy"
        assert np.array_equal(np.load(path), np.array([0, 0], dtype=np.uint8))
