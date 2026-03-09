"""
NGC628 completeness prediction: single-call API for programmatic use.

Use:
    from completeness_nn_api import ngc628_completeness_predict as predict
    predict(phys=(mass, age, av))   # arrays of shape (n,) or (n,3)
    predict(phot=mag_5band)        # array of shape (n, 5)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .inference import load_model_and_scalers

# Lazy-loaded models (same as serve.py)
_models = None
# Default: bundled checkpoints inside the package (so pip install is enough)
_PKG_CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
_default_model_dir = Path(
    os.environ.get("COMPLETENESS_MODEL_DIR", str(_PKG_CHECKPOINTS))
).resolve()
_default_outname = os.environ.get("COMPLETENESS_OUTNAME", "model0")


def _get_models(model_dir: Path | None = None, outname: str | None = None):
    global _models
    if _models is None:
        d = Path(model_dir).resolve() if model_dir is not None else _default_model_dir
        n = outname if outname is not None else _default_outname
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _models = (*load_model_and_scalers(d, n, device=device), device)
    return _models


def ngc628_completeness_predict(
    *,
    phys: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    phot: np.ndarray | None = None,
    model_dir: Path | str | None = None,
    outname: str | None = None,
) -> np.ndarray:
    """
    Predict detection completeness for NGC628-like completeness model.

    Provide either phys or phot (not both in one call).

    Parameters
    ----------
    phys : array of shape (n, 3) or tuple (mass, age, av) each (n,)
        Physical features: mass [M_sun], age [yr], av [mag].
    phot : array of shape (n, 5)
        Five-band magnitudes (e.g. F275W, F336W, F438W, F555W, F814W).
    model_dir : path, optional
        Directory with the four checkpoint files. Default from env COMPLETENESS_MODEL_DIR or "checkpoints".
    outname : str, optional
        Model suffix (default from env COMPLETENESS_OUTNAME or "model0").

    Returns
    -------
    np.ndarray of shape (n,)
        Completeness probability in [0, 1] per source.
    """
    if (phys is None) == (phot is None):
        raise ValueError("Provide exactly one of phys= or phot=")

    model_phys, model_phot, scaler_phys, scaler_phot, device = _get_models(
        Path(model_dir) if model_dir is not None else None, outname
    )

    if phys is not None:
        if isinstance(phys, (tuple, list)) and len(phys) == 3:
            x = np.column_stack([np.asarray(phys[0]), np.asarray(phys[1]), np.asarray(phys[2])])
        else:
            x = np.asarray(phys, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("phys must be (n, 3) or (mass, age, av) each (n,)")
        x_scaled = scaler_phys.transform(x)
        model = model_phys
    else:
        x = np.asarray(phot, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError("phot must be (n, 5)")
        x_scaled = scaler_phot.transform(x)
        model = model_phot

    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_t).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


# Alias for shorter import
predict = ngc628_completeness_predict
