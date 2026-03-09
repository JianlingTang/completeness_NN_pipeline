"""
Tests for cluster_pipeline.data.models.
Focus: cluster_id integrity, data alignment, detection labels consistency.
"""
from pathlib import Path

from cluster_pipeline.data.models import (
    DetectionLabelRecord,
    MatchResult,
    SyntheticCluster,
)


class TestSyntheticCluster:
    """SyntheticCluster: cluster_id required, to_injection_row format."""

    def test_creation_with_cluster_id(self):
        c = SyntheticCluster(
            mass=1e4,
            age=1e8,
            av=0.2,
            radius=10.0,
            position=(100.5, 200.3),
            cluster_id=42,
        )
        assert c.cluster_id == 42
        assert c.position == (100.5, 200.3)

    def test_to_injection_row_default_mag_column(self):
        c = SyntheticCluster(
            mass=1e4,
            age=1e8,
            av=0.2,
            radius=10.0,
            position=(10.0, 20.0),
            cluster_id=1,
            photometry=[17.0, 18.0, 19.0],
        )
        # mag_column=2 -> photometry[2] = 19.0
        row = c.to_injection_row(mag_column=2)
        assert "10.00 20.00 19.0000" in row
        assert row.endswith("\n")

    def test_to_injection_row_no_photometry_uses_zero(self):
        c = SyntheticCluster(
            mass=1e4,
            age=1e8,
            av=0.2,
            radius=10.0,
            position=(10.0, 20.0),
            cluster_id=1,
            photometry=None,
        )
        row = c.to_injection_row(mag_column=0)
        assert "10.00 20.00 0.0000" in row


class TestMatchResult:
    """MatchResult: cluster_id alignment with matched_indices, detection_labels."""

    def test_get_matched_cluster_ids_order(self):
        mr = MatchResult(
            injected_path=Path("."),
            detected_path=Path("."),
            cluster_ids=[10, 20, 30, 40],
            matched_indices=[0, 2],
            matched_positions=[(1.0, 2.0), (3.0, 4.0)],
            n_injected=4,
            n_matched=2,
            tolerance_pix=3.0,
        )
        assert mr.get_matched_cluster_ids() == [10, 30]

    def test_detection_labels_binary_per_injected(self):
        mr = MatchResult(
            injected_path=Path("."),
            detected_path=Path("."),
            cluster_ids=[10, 20, 30],
            matched_indices=[1],
            matched_positions=[(5.0, 6.0)],
            n_injected=3,
            n_matched=1,
            tolerance_pix=3.0,
        )
        assert mr.detection_labels == [0, 1, 0]

    def test_detection_label_by_cluster_id(self):
        mr = MatchResult(
            injected_path=Path("."),
            detected_path=Path("."),
            cluster_ids=[10, 20, 30],
            matched_indices=[0, 2],
            matched_positions=[(1.0, 2.0), (3.0, 4.0)],
            n_injected=3,
            n_matched=2,
            tolerance_pix=3.0,
        )
        d = mr.detection_label_by_cluster_id()
        assert d == {10: 1, 20: 0, 30: 1}

    def test_detection_labels_length_equals_n_injected(self):
        mr = MatchResult(
            injected_path=Path("."),
            detected_path=Path("."),
            cluster_ids=[1, 2, 3, 4, 5],
            matched_indices=[0, 4],
            matched_positions=[(0.0, 0.0), (1.0, 1.0)],
            n_injected=5,
            n_matched=2,
            tolerance_pix=3.0,
        )
        assert len(mr.detection_labels) == mr.n_injected
        assert sum(mr.detection_labels) == mr.n_matched


class TestDetectionLabelRecord:
    """DetectionLabelRecord: one row per cluster/frame/reff."""

    def test_record_fields(self):
        r = DetectionLabelRecord(
            cluster_id=7,
            frame_id=2,
            reff_id=0,
            reff=5.0,
            detected=1,
            galaxy_id="NGC123",
        )
        assert r.cluster_id == 7
        assert r.detected in (0, 1)
