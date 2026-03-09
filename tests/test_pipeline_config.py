"""
Tests for cluster_pipeline.config.pipeline_config.
Focus: config structure, path helpers, overrides.
"""
from pathlib import Path

from cluster_pipeline.config.pipeline_config import (
    PipelineConfig,
    get_config,
)


class TestPipelineConfig:
    """PipelineConfig: frozen dataclass, path helpers."""

    def test_galaxy_dir(self):
        cfg = PipelineConfig(
            main_dir=Path("/main"),
            fits_path=Path("/fits"),
            psf_path=Path("/psf"),
            bao_path=Path("/bao"),
            slug_lib_dir=Path("/slug"),
            output_lib_dir=Path("/out"),
            temp_base_dir=Path("/tmp"),
        )
        assert cfg.galaxy_dir("NGC123") == Path("/main/NGC123")

    def test_white_dir(self):
        cfg = PipelineConfig(
            main_dir=Path("/main"),
            fits_path=Path("/fits"),
            psf_path=Path("/psf"),
            bao_path=Path("/bao"),
            slug_lib_dir=Path("/slug"),
            output_lib_dir=Path("/out"),
            temp_base_dir=Path("/tmp"),
        )
        assert "white" in str(cfg.white_dir("NGC123"))

    def test_temp_dir_for(self):
        cfg = PipelineConfig(
            main_dir=Path("/main"),
            fits_path=Path("/fits"),
            psf_path=Path("/psf"),
            bao_path=Path("/bao"),
            slug_lib_dir=Path("/slug"),
            output_lib_dir=Path("/out"),
            temp_base_dir=Path("/tmp"),
        )
        p = cfg.temp_dir_for("G1", 2, 10.0)
        assert "G1" in str(p)
        assert "frame2" in str(p)
        assert "reff10" in str(p)

    def test_default_thres_coord(self):
        cfg = PipelineConfig(
            main_dir=Path("/m"),
            fits_path=Path("/f"),
            psf_path=Path("/p"),
            bao_path=Path("/b"),
            slug_lib_dir=Path("/s"),
            output_lib_dir=Path("/o"),
            temp_base_dir=Path("/t"),
        )
        assert cfg.thres_coord == 3.0


class TestGetConfigAndOverrides:
    """get_config with overrides; _apply_overrides."""

    def test_get_config_returns_config(self):
        cfg = get_config(overrides=None)
        assert isinstance(cfg, PipelineConfig)
        assert hasattr(cfg, "main_dir")
        assert hasattr(cfg, "thres_coord")

    def test_overrides_apply(self):
        cfg = get_config(overrides={"thres_coord": 5.0})
        assert cfg.thres_coord == 5.0

    def test_overrides_main_dir_path(self):
        cfg = get_config(overrides={"main_dir": "/custom/main"})
        assert "custom" in str(cfg.main_dir) or "main" in str(cfg.main_dir)
