"""
Tests for cluster_pipeline.matching.coordinate_matcher.
Focus: coordinate matching correctness, cluster_id integrity, load_coords alignment.
"""
from pathlib import Path

import numpy as np
import pytest

from cluster_pipeline.data.models import MatchResult
from cluster_pipeline.matching.coordinate_matcher import (
    CoordinateMatcher,
    load_coords,
    match_coordinates,
)


class TestLoadCoords:
    """load_coords: two-column (x,y) or three-column (x y mag) -> (N,2) float64."""

    def test_two_column_file(self, sample_injected_coords):
        arr = load_coords(sample_injected_coords)
        assert arr.shape[1] == 2
        assert arr.dtype == np.float64
        assert arr.shape[0] == 3
        np.testing.assert_array_almost_equal(arr[0], [10.0, 20.0])

    def test_three_column_returns_only_xy(self, sample_coords_three_col):
        arr = load_coords(sample_coords_three_col)
        assert arr.shape == (2, 2)
        np.testing.assert_array_almost_equal(arr[0], [1.0, 2.0])

    def test_single_row_reshape(self, tmp_path_dir):
        p = tmp_path_dir / "single.coo"
        p.write_text("1.0 2.0\n")
        arr = load_coords(p)
        assert arr.shape == (1, 2)
        np.testing.assert_array_almost_equal(arr[0], [1.0, 2.0])


class TestMatchCoordinates:
    """match_coordinates: KD-tree match, cluster_ids aligned with injected order."""

    def test_empty_injected_returns_zero_matched(self, sample_detected_coords, sample_cluster_ids):
        injected = np.array([]).reshape(0, 2)
        detected = np.loadtxt(sample_detected_coords)
        res = match_coordinates(
            injected,
            detected,
            tolerance_pix=3.0,
            cluster_ids=[],
        )
        assert res.n_injected == 0
        assert res.n_matched == 0
        assert res.matched_indices == []
        assert res.matched_positions == []

    def test_empty_detected_returns_zero_matched(self, sample_injected_coords, sample_cluster_ids):
        injected = np.loadtxt(sample_injected_coords)
        detected = np.array([]).reshape(0, 2)
        res = match_coordinates(
            injected,
            detected,
            tolerance_pix=3.0,
            cluster_ids=sample_cluster_ids,
        )
        assert res.n_injected == 3
        assert res.n_matched == 0
        assert res.get_matched_cluster_ids() == []

    def test_full_match_within_tolerance(self):
        # Same positions -> all match
        coords = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        cids = [100, 200]
        res = match_coordinates(coords, coords, tolerance_pix=1.0, cluster_ids=cids)
        assert res.n_matched == 2
        assert res.get_matched_cluster_ids() == [100, 200]
        assert res.detection_labels == [1, 1]

    def test_partial_match_cluster_id_integrity(
        self,
        sample_injected_coords,
        sample_detected_coords,
        sample_cluster_ids,
    ):
        # Injected: (10,20), (30,40), (50,60). Detected: (11,21), (31,41), (100,200).
        # First two within 3 px, third not.
        res = match_coordinates(
            sample_injected_coords,
            sample_detected_coords,
            tolerance_pix=3.0,
            cluster_ids=sample_cluster_ids,
        )
        assert res.n_injected == 3
        assert res.n_matched == 2
        assert res.get_matched_cluster_ids() == [101, 102]
        assert res.detection_labels == [1, 1, 0]
        assert res.detection_label_by_cluster_id() == {101: 1, 102: 1, 103: 0}

    def test_cluster_ids_length_mismatch_raises(self, sample_injected_coords):
        injected = np.loadtxt(sample_injected_coords)
        with pytest.raises(ValueError, match="cluster_ids length must match"):
            match_coordinates(
                injected,
                injected,
                tolerance_pix=5.0,
                cluster_ids=[1, 2],
            )

    def test_cluster_ids_default_is_range_n(self, sample_injected_coords, sample_detected_coords):
        res = match_coordinates(
            sample_injected_coords,
            sample_detected_coords,
            tolerance_pix=3.0,
            cluster_ids=None,
        )
        assert res.cluster_ids == [0, 1, 2]
        assert res.get_matched_cluster_ids() == [0, 1]

    def test_path_inputs(self, sample_injected_coords, sample_detected_coords, sample_cluster_ids):
        res = match_coordinates(
            sample_injected_coords,
            sample_detected_coords,
            tolerance_pix=3.0,
            cluster_ids=sample_cluster_ids,
        )
        assert res.injected_path == sample_injected_coords
        assert res.detected_path == sample_detected_coords
        assert res.n_matched == 2


class TestCoordinateMatcher:
    """CoordinateMatcher wrapper uses tolerance and returns MatchResult."""

    def test_match_delegates_with_tolerance(self, sample_injected_coords, sample_detected_coords):
        # Use same coords for injected and detected so with large tolerance all match
        coords = np.loadtxt(sample_injected_coords)
        matcher = CoordinateMatcher(tolerance_pix=10.0)
        res = matcher.match(coords, coords, cluster_ids=[1, 2, 3])
        assert res.n_matched == 3
        assert res.tolerance_pix == 10.0

    def test_write_matched_coords_empty(self, tmp_path_dir, sample_detected_coords):
        matcher = CoordinateMatcher(tolerance_pix=3.0)
        res = MatchResult(
            injected_path=Path("."),
            detected_path=sample_detected_coords,
            cluster_ids=[],
            matched_indices=[],
            matched_positions=[],
            n_injected=0,
            n_matched=0,
            tolerance_pix=3.0,
        )
        out = tmp_path_dir / "out.coo"
        matcher.write_matched_coords(res, out, sample_detected_coords)
        assert out.read_text() == ""

    def test_write_matched_coords_writes_xy(self, tmp_path_dir):
        matcher = CoordinateMatcher(tolerance_pix=3.0)
        res = MatchResult(
            injected_path=Path("."),
            detected_path=Path("."),
            cluster_ids=[1, 2],
            matched_indices=[0, 1],
            matched_positions=[(11.0, 21.0), (31.0, 41.0)],
            n_injected=2,
            n_matched=2,
            tolerance_pix=3.0,
        )
        detected_path = tmp_path_dir / "det.coo"
        np.savetxt(detected_path, np.array([[11.0, 21.0], [31.0, 41.0]]), fmt="%.2f")
        out = tmp_path_dir / "matched.coo"
        matcher.write_matched_coords(res, out, detected_path, include_mag=False)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert "11.00 21.00" in lines[0]
        assert "31.00 41.00" in lines[1]
