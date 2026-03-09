"""
Tests for cluster_pipeline.pipeline.manifest.
Focus: job status consistency, (galaxy_id, frame_id, reff) key integrity.
"""
from pathlib import Path

from cluster_pipeline.pipeline.manifest import (
    MANIFEST_SCHEMA,
    STATUS_FAILED,
    STATUS_MATCHING_DONE,
    STATUS_PENDING,
    get_job_status,
    list_pending_jobs,
    load_manifest,
    manifest_path,
    save_manifest,
    set_job_status,
)


class TestLoadSaveManifest:
    """load_manifest / save_manifest: empty DataFrame when missing, roundtrip."""

    def test_load_missing_returns_empty_dataframe(self, tmp_path_dir):
        path = tmp_path_dir / "nonexistent.parquet"
        df = load_manifest(path)
        assert df.empty
        assert list(df.columns) == list(MANIFEST_SCHEMA.keys())

    def test_save_and_load_roundtrip(self, tmp_path_dir):
        import pandas as pd
        path = tmp_path_dir / "manifest.parquet"
        df = pd.DataFrame([{
            "galaxy_id": "NGC123",
            "frame_id": 0,
            "reff": 5.0,
            "status": "pending",
            "outname": "pipeline",
            "updated_at": "2025-01-01T00:00:00Z",
        }])
        save_manifest(df, path)
        assert path.exists()
        back = load_manifest(path)
        assert len(back) == 1
        assert back["galaxy_id"].iloc[0] == "NGC123"
        assert back["frame_id"].iloc[0] == 0
        assert back["reff"].iloc[0] == 5.0


class TestGetSetJobStatus:
    """get_job_status / set_job_status: key = (galaxy_id, frame_id, reff)."""

    def test_set_then_get(self, tmp_path_dir):
        path = tmp_path_dir / "manifest.parquet"
        set_job_status(path, "G1", 1, 10.0, STATUS_MATCHING_DONE)
        status = get_job_status(path, "G1", 1, 10.0)
        assert status == STATUS_MATCHING_DONE

    def test_get_missing_returns_none(self, tmp_path_dir):
        path = tmp_path_dir / "manifest.parquet"
        save_manifest(__import__("pandas").DataFrame(columns=list(MANIFEST_SCHEMA.keys())), path)
        assert get_job_status(path, "X", 99, 7.0) is None

    def test_set_updates_existing_row(self, tmp_path_dir):
        path = tmp_path_dir / "manifest.parquet"
        set_job_status(path, "G1", 0, 5.0, "pending")
        set_job_status(path, "G1", 0, 5.0, STATUS_MATCHING_DONE)
        assert get_job_status(path, "G1", 0, 5.0) == STATUS_MATCHING_DONE


class TestListPendingJobs:
    """list_pending_jobs: returns (frame_id, reff) for pending or failed."""

    def test_empty_manifest_returns_empty_list(self, tmp_path_dir):
        path = tmp_path_dir / "manifest.parquet"
        save_manifest(__import__("pandas").DataFrame(columns=list(MANIFEST_SCHEMA.keys())), path)
        jobs = list_pending_jobs(path, "G1")
        assert jobs == []

    def test_returns_only_pending_or_failed_for_galaxy(self, tmp_path_dir):
        path = tmp_path_dir / "manifest.parquet"
        set_job_status(path, "G1", 0, 5.0, STATUS_PENDING)
        set_job_status(path, "G1", 1, 10.0, STATUS_FAILED)
        set_job_status(path, "G1", 2, 15.0, STATUS_MATCHING_DONE)
        jobs = list_pending_jobs(path, "G1")
        assert (0, 5.0) in jobs
        assert (1, 10.0) in jobs
        assert (2, 15.0) not in jobs


class TestManifestPath:
    """manifest_path: path construction."""

    def test_returns_path_under_config_galaxy_white(self):
        base = Path("/base")
        p = manifest_path(base, "NGC123", outname="pipeline")
        assert "NGC123" in str(p)
        assert "white" in str(p)
        assert "manifest_pipeline.parquet" in str(p)
