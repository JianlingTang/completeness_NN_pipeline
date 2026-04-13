"""
SExtractor runner: encapsulate SExtractor invocation.
Runs in a given working directory; no os.chdir in caller.
Returns paths to catalog and .coo file for matching.
"""
import subprocess
from pathlib import Path

from ..config import PipelineConfig
from ..data.models import DetectionResult


def _read_config_value(config_path: Path, key: str) -> str | None:
    """Extract a single value from a SExtractor config file."""
    try:
        for line in Path(config_path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line.upper().startswith(key.upper()):
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    return parts[1].strip()
    except Exception:
        pass
    return None


def _find_sextractor_share() -> Path | None:
    """Locate SExtractor's share directory (Homebrew or system)."""
    candidates = [
        Path("/opt/homebrew/Cellar/sextractor"),
        Path("/usr/local/Cellar/sextractor"),
        Path("/usr/share/sextractor"),
        Path("/usr/local/share/sextractor"),
    ]
    for base in candidates:
        if base.is_dir():
            if (base / "default.conv").exists():
                return base
            for child in sorted(base.iterdir(), reverse=True):
                share = child / "share" / "sextractor"
                if share.is_dir():
                    return share
    return None


def run_sextractor(
    frame_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    param_path: Path | None = None,
    nnw_path: Path | None = None,
    catalog_name: str = "detection.cat",
    coo_suffix: str = ".coo",
) -> DetectionResult:
    """
    Run SExtractor on a single frame. Writes catalog and coord file into output_dir.

    Parameters
    ----------
    frame_path : Path
        Path to the FITS frame.
    output_dir : Path
        Directory for output catalog and .coo file; must exist.
    config_path : Path, optional
        Path to SExtractor config file (-c).
    param_path : Path, optional
        Path to parameter file (PARAMETERS_NAME in config).
    nnw_path : Path, optional
        Path to neural network weights (FILTER_NAME in config).
    catalog_name : str
        Output catalog filename.
    coo_suffix : str
        Suffix for coord file derived from frame stem (stem + coo_suffix).

    Returns
    -------
    DetectionResult
        catalog_path, coord_path, n_detected (from catalog line count), frame_path.
    """
    output_dir = Path(output_dir).resolve()
    frame_path = Path(frame_path).resolve()
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["sex", str(frame_path)]
    if config_path is not None:
        config_path = Path(config_path).resolve()
        if config_path.exists():
            cmd.extend(["-c", str(config_path)])
    if param_path is not None and Path(param_path).exists():
        cmd.extend(["-PARAMETERS_NAME", str(Path(param_path).resolve())])
    if nnw_path is not None and Path(nnw_path).exists():
        cmd.extend(["-STARNNW_NAME", str(Path(nnw_path).resolve())])
    cmd.extend(["-CATALOG_NAME", str(output_dir / catalog_name)])
    # Force detection thresholds regardless of values in the config file.
    cmd.extend([
        "-DETECT_MINAREA", "5",
        "-DETECT_THRESH", "10",
        "-ANALYSIS_THRESH", "10",
    ])

    # Resolve FILTER_NAME (convolution file) from SExtractor's share dir if needed
    if config_path is not None and config_path.exists():
        _filter_name = _read_config_value(config_path, "FILTER_NAME")
        if _filter_name and not Path(_filter_name).is_absolute():
            _sex_share = _find_sextractor_share()
            if _sex_share:
                _conv = _sex_share / _filter_name
                if _conv.exists():
                    cmd.extend(["-FILTER_NAME", str(_conv)])
    # Run from output_dir so relative paths in config resolve there
    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SExtractor failed (exit {result.returncode}): {result.stderr or result.stdout}"
        )

    catalog_path = output_dir / catalog_name
    coo_path = output_dir / f"{frame_path.stem}{coo_suffix}"
    # If config writes a different catalog name, look for it
    if not catalog_path.exists():
        cats = list(output_dir.glob("*.cat"))
        catalog_path = cats[0] if cats else catalog_path

    n_detected = 0
    if catalog_path.exists():
        with open(catalog_path) as f:
            n_detected = sum(1 for line in f if not line.startswith("#"))
        n_detected = max(0, n_detected)

    # Always build .coo from current catalog so matching uses this run's detections (overwrite any stale .coo)
    if catalog_path.exists():
        _write_coo_from_catalog(catalog_path, coo_path, param_path=param_path)

    return DetectionResult(
        catalog_path=catalog_path,
        coord_path=coo_path,
        n_detected=n_detected,
        frame_path=frame_path,
    )


def _write_coo_from_catalog(
    catalog_path: Path, coo_path: Path, param_path: Path | None = None
) -> None:
    """Write two-column (x, y) = (X_IMAGE, Y_IMAGE) from SExtractor catalog.
    Column order is taken from PARAMETERS_NAME (param_path) so we output (x, y) for matching.
    """
    import numpy as np
    try:
        data = np.loadtxt(catalog_path, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            return
        # Resolve (x, y) column indices from param file (same order as SExtractor catalog)
        xcol, ycol = _x_y_column_indices(param_path)
        xy = data[:, [xcol, ycol]]
        np.savetxt(coo_path, xy, fmt="%.4f")
    except Exception:
        coo_path.write_text("")


def _x_y_column_indices(param_path: Path | None) -> tuple[int, int]:
    """Return (col_index_X_IMAGE, col_index_Y_IMAGE) from PARAMETERS_NAME file.
    If param_path is missing or unreadable, assume first two columns are (X, Y).
    """
    if param_path is None or not Path(param_path).exists():
        return 0, 1
    col = 0
    xcol, ycol = None, None
    for line in Path(param_path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.upper() == "X_IMAGE":
            xcol = col
        elif line.upper() == "Y_IMAGE":
            ycol = col
        col += 1
        if xcol is not None and ycol is not None:
            return xcol, ycol
    return 0, 1


class SExtractorRunner:
    """
    SExtractor runner with config. Holds paths from PipelineConfig;
    run() executes on one frame and returns DetectionResult.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        frame_path: Path,
        output_dir: Path,
        catalog_name: str = "detection.cat",
        coo_suffix: str = ".coo",
    ) -> DetectionResult:
        return run_sextractor(
            frame_path=frame_path,
            output_dir=output_dir,
            config_path=self.config.sextractor_config_path,
            param_path=self.config.sextractor_param_path,
            nnw_path=self.config.sextractor_nnw_path,
            catalog_name=catalog_name,
            coo_suffix=coo_suffix,
        )
