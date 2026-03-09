"""
Pipeline configuration: single source of truth for paths, sampling, matching, and detection.
No hardcoded paths in pipeline code; load from env with fallback to project root.
"""
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Repo root (cluster_pipeline/config/ -> parent.parent.parent)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _p(path: str | Path) -> Path:
    return Path(path).resolve()


@dataclass(frozen=True)
class PipelineConfig:
    """
    All configurable parameters for the cluster completeness pipeline.
    Paths are absolute or relative to a configured root; pipeline uses pathlib.
    """

    main_dir: Path
    fits_path: Path
    psf_path: Path
    bao_path: Path
    slug_lib_dir: Path
    output_lib_dir: Path
    temp_base_dir: Path
    make_legus_cct_dir: Path | None = None

    ncl: int = 500
    nframe: int = 50
    reff_list: list[float] = field(default_factory=lambda: [float(r) for r in range(1, 11)])
    mrmodel: str = "flat"
    dmod: float = 29.98
    M_LIMIT: float = 15.0
    thres_coord: float = 3.0
    validation: bool = False
    overwrite: bool = False
    sextractor_config_path: Path | None = None
    sextractor_param_path: Path | None = None
    sextractor_nnw_path: Path | None = None
    pixscale_wfc3: float = 0.04
    pixscale_acs: float = 0.05
    sigma_pc: float = 100.0
    merr_cut: float = 0.3
    inject_5filter_script: Path | None = None

    def galaxy_dir(self, galaxy_id: str) -> Path:
        return self.main_dir / galaxy_id

    def white_dir(self, galaxy_id: str) -> Path:
        return self.galaxy_dir(galaxy_id) / "white"

    def synthetic_fits_dir(self, galaxy_id: str) -> Path:
        return self.white_dir(galaxy_id) / "synthetic_fits"

    def s_extraction_dir(self, galaxy_id: str) -> Path:
        return self.white_dir(galaxy_id) / "s_extraction"

    def matched_coords_dir(self, galaxy_id: str) -> Path:
        return self.white_dir(galaxy_id) / "matched_coords"

    def diagnostics_dir(self, galaxy_id: str) -> Path:
        return self.white_dir(galaxy_id) / "diagnostics"

    def detection_labels_dir(self, galaxy_id: str) -> Path:
        return self.white_dir(galaxy_id) / "detection_labels"

    def filter_synthetic_fits_dir(self, galaxy_id: str, filter_name: str) -> Path:
        return self.galaxy_dir(galaxy_id) / filter_name / "synthetic_fits"

    def photometry_dir(self, galaxy_id: str, filter_name: str) -> Path:
        return self.galaxy_dir(galaxy_id) / filter_name / "photometry"

    def catalogue_dir(self, galaxy_id: str) -> Path:
        return self.white_dir(galaxy_id) / "catalogue"

    def physprop_dir(self) -> Path:
        return self.main_dir / "physprop"

    def temp_dir_for(self, galaxy_id: str, frame_id: int, reff: float) -> Path:
        return self.temp_base_dir / f"{galaxy_id}_frame{frame_id}_reff{reff}"


def get_config(overrides: dict | None = None) -> PipelineConfig:
    """
    Build PipelineConfig from defaults (project root), then env, then overrides.
    Env keys: COMP_MAIN_DIR, COMP_FITS_PATH, COMP_PSF_PATH, COMP_BAO_PATH,
    COMP_SLUG_LIB_DIR, COMP_OUTPUT_LIB_DIR, COMP_TEMP_BASE_DIR.
    When unset, paths default to project root (e.g. main_dir=PROJECT_ROOT).
    """
    def _path(key: str, default: Path) -> Path:
        raw = os.environ.get(key)
        if raw:
            return _p(raw)
        return _p(default)

    main_dir = _path("COMP_MAIN_DIR", _PROJECT_ROOT)
    fits_path = _path("COMP_FITS_PATH", main_dir)
    psf_path = _path("COMP_PSF_PATH", _PROJECT_ROOT / "PSF_all")
    bao_path = _path("COMP_BAO_PATH", _PROJECT_ROOT / "baolab")
    slug_lib_dir = _path("COMP_SLUG_LIB_DIR", _PROJECT_ROOT / "SLUG_library")
    output_lib_dir = _path("COMP_OUTPUT_LIB_DIR", _PROJECT_ROOT / "output_lib")
    temp_base_dir = _path("COMP_TEMP_BASE_DIR", Path(tempfile.gettempdir()) / "cluster_pipeline")
    make_legus_cct_dir = os.environ.get("COMP_MAKE_LEGUS_CCT_DIR")
    if make_legus_cct_dir:
        make_legus_cct_dir = _p(make_legus_cct_dir)

    cfg = PipelineConfig(
        main_dir=main_dir,
        fits_path=fits_path,
        psf_path=psf_path,
        bao_path=bao_path,
        slug_lib_dir=slug_lib_dir,
        output_lib_dir=output_lib_dir,
        temp_base_dir=temp_base_dir,
        make_legus_cct_dir=make_legus_cct_dir,
    )
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg: PipelineConfig, overrides: dict) -> PipelineConfig:
    """Return a new config with overrides applied (only simple/Path fields)."""
    d = {
        "main_dir": cfg.main_dir, "fits_path": cfg.fits_path, "psf_path": cfg.psf_path,
        "bao_path": cfg.bao_path, "slug_lib_dir": cfg.slug_lib_dir,
        "output_lib_dir": cfg.output_lib_dir, "temp_base_dir": cfg.temp_base_dir,
        "make_legus_cct_dir": cfg.make_legus_cct_dir,
        "ncl": cfg.ncl, "nframe": cfg.nframe, "reff_list": list(cfg.reff_list),
        "mrmodel": cfg.mrmodel, "dmod": cfg.dmod, "M_LIMIT": cfg.M_LIMIT,
        "thres_coord": cfg.thres_coord, "validation": cfg.validation, "overwrite": cfg.overwrite,
        "inject_5filter_script": cfg.inject_5filter_script,
        "sextractor_config_path": cfg.sextractor_config_path,
        "sextractor_param_path": cfg.sextractor_param_path,
        "sextractor_nnw_path": cfg.sextractor_nnw_path,
        "pixscale_wfc3": cfg.pixscale_wfc3, "pixscale_acs": cfg.pixscale_acs,
        "sigma_pc": cfg.sigma_pc, "merr_cut": cfg.merr_cut,
    }
    for k, v in overrides.items():
        if k in d:
            if k in ("main_dir", "fits_path", "psf_path", "bao_path", "slug_lib_dir",
                     "output_lib_dir", "temp_base_dir", "make_legus_cct_dir") and v is not None:
                d[k] = _p(str(v))
            elif k == "inject_5filter_script" and v is not None:
                d[k] = _p(str(v))
            else:
                d[k] = v
    return PipelineConfig(**d)
