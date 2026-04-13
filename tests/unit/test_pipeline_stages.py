"""Unit tests for cluster_pipeline.pipeline.stages."""
import pytest

from cluster_pipeline.pipeline.stages import (
    LAST_STAGE,
    STAGE_CATALOGUE,
    STAGE_DATASET,
    STAGE_DETECTION,
    STAGE_INJECTION,
    STAGE_MATCHING,
    STAGE_NAMES,
    STAGE_PHOTOMETRY,
    run_stage,
)


class TestStageConstants:
    """Stage numbers and names are consistent."""

    def test_stage_numbers_sequential(self):
        assert STAGE_INJECTION == 1
        assert STAGE_DETECTION == 2
        assert STAGE_MATCHING == 3
        assert STAGE_PHOTOMETRY == 4
        assert STAGE_CATALOGUE == 5
        assert STAGE_DATASET == 6

    def test_last_stage(self):
        assert LAST_STAGE == STAGE_DATASET

    def test_stage_names_cover_all_stages(self):
        for stage in (STAGE_INJECTION, STAGE_DETECTION, STAGE_MATCHING,
                      STAGE_PHOTOMETRY, STAGE_CATALOGUE, STAGE_DATASET):
            assert stage in STAGE_NAMES
            assert isinstance(STAGE_NAMES[stage], str)
            assert len(STAGE_NAMES[stage]) > 0

    def test_stage_names_expected_values(self):
        assert STAGE_NAMES[STAGE_INJECTION] == "injection"
        assert STAGE_NAMES[STAGE_DETECTION] == "detection"
        assert STAGE_NAMES[STAGE_MATCHING] == "matching"
        assert STAGE_NAMES[STAGE_PHOTOMETRY] == "photometry"
        assert STAGE_NAMES[STAGE_CATALOGUE] == "catalogue"
        assert STAGE_NAMES[STAGE_DATASET] == "dataset"


class TestRunStage:
    """run_stage(stage, max_stage) determines if stage should run."""

    def test_max_stage_none_always_run(self):
        """When max_stage is None, all stages run (caller uses flags)."""
        assert run_stage(STAGE_INJECTION, None) is True
        assert run_stage(STAGE_DATASET, None) is True

    def test_max_stage_1_only_injection(self):
        assert run_stage(STAGE_INJECTION, 1) is True
        assert run_stage(STAGE_DETECTION, 1) is False
        assert run_stage(STAGE_MATCHING, 1) is False
        assert run_stage(STAGE_DATASET, 1) is False

    def test_max_stage_3_injection_detection_matching(self):
        assert run_stage(STAGE_INJECTION, 3) is True
        assert run_stage(STAGE_DETECTION, 3) is True
        assert run_stage(STAGE_MATCHING, 3) is True
        assert run_stage(STAGE_PHOTOMETRY, 3) is False
        assert run_stage(STAGE_CATALOGUE, 3) is False
        assert run_stage(STAGE_DATASET, 3) is False

    def test_max_stage_6_all_run(self):
        for stage in (STAGE_INJECTION, STAGE_DETECTION, STAGE_MATCHING,
                      STAGE_PHOTOMETRY, STAGE_CATALOGUE, STAGE_DATASET):
            assert run_stage(stage, 6) is True

    def test_stage_below_range(self):
        """Stage 0 or negative: run_stage treats as not in 1..max_stage."""
        assert run_stage(0, 3) is False
        assert run_stage(-1, 5) is False

    def test_stage_above_max_stage(self):
        assert run_stage(STAGE_DATASET, 5) is False
        assert run_stage(7, 6) is False
