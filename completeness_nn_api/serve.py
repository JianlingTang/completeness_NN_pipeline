"""
External HTTP API for cluster completeness prediction.

Run with:
  COMPLETENESS_MODEL_DIR=/path/to/checkpoints COMPLETENESS_OUTNAME=model0 uvicorn completeness_nn_api.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import load_model_and_scalers, MLP

# ---- Config from env (default: bundled checkpoints inside the package) ----
_PKG_CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
MODEL_DIR = Path(
    os.environ.get("COMPLETENESS_MODEL_DIR", str(_PKG_CHECKPOINTS))
).resolve()
OUTNAME = os.environ.get("COMPLETENESS_OUTNAME", "model0")

app = FastAPI(
    title="Cluster Completeness API",
    description="Predict detection completeness from physical (mass, age, av) or photometric (5-band mag) features using a trained MLP.",
    version="0.1.0",
)

# Loaded at startup (fail fast if artifacts missing)
_model_phys: MLP | None = None
_model_phot: MLP | None = None
_scaler_phys = None
_scaler_phot = None
_device: torch.device | None = None


@app.on_event("startup")
def startup():
    global _model_phys, _model_phot, _scaler_phys, _scaler_phot, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model_phys, _model_phot, _scaler_phys, _scaler_phot = load_model_and_scalers(
        MODEL_DIR, OUTNAME, device=_device
    )


# ---- Request/response schemas ----

class PredictPhysRequest(BaseModel):
    """Physical features: mass (M_sun), age (yr), av (mag). Same units as training."""
    mass: list[float] = Field(..., description="Mass in M_sun")
    age: list[float] = Field(..., description="Age in yr")
    av: list[float] = Field(..., description="A_V in mag")

    def to_array(self) -> np.ndarray:
        n = len(self.mass)
        if len(self.age) != n or len(self.av) != n:
            raise ValueError("mass, age, av must have the same length")
        return np.column_stack([self.mass, self.age, self.av]).astype(np.float64)


class PredictPhotRequest(BaseModel):
    """Photometric features: 5-band magnitudes (e.g. F275W, F336W, F438W, F555W, F814W). Same order as training (mag_f0..mag_f4)."""
    phot: list[list[float]] = Field(
        ...,
        description="List of [mag_f0, mag_f1, mag_f2, mag_f3, mag_f4] per source"
    )

    def to_array(self) -> np.ndarray:
        return np.asarray(self.phot, dtype=np.float64)


class PredictResponse(BaseModel):
    """Completeness probability in [0, 1] per input row."""
    completeness: list[float] = Field(..., description="P(detected) per source")


# ---- Endpoints ----

@app.get("/health")
def health():
    """Liveness/readiness: returns 200 if models are loaded."""
    if _model_phys is None or _model_phot is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok", "device": str(_device)}


@app.post("/predict_phys", response_model=PredictResponse)
def predict_phys(req: PredictPhysRequest):
    """
    Predict completeness from physical properties (mass, age, av).
    Input units: mass [M_sun], age [yr], av [mag]. Same as training data.
    """
    try:
        x = req.to_array()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if x.shape[1] != 3:
        raise HTTPException(status_code=400, detail="Expected 3 columns: mass, age, av")
    x_scaled = _scaler_phys.transform(x)
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=_device)
    with torch.no_grad():
        logits = _model_phys(x_t).squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy().tolist()
    return PredictResponse(completeness=probs)


@app.post("/predict_phot", response_model=PredictResponse)
def predict_phot(req: PredictPhotRequest):
    """
    Predict completeness from 5-band magnitudes (mag_f0..mag_f4).
    Same filter order as training (e.g. F275W, F336W, F438W, F555W, F814W).
    """
    x = np.asarray(req.phot, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 5:
        raise HTTPException(
            status_code=400,
            detail="Expected shape (n_samples, 5) for phot (mag_f0..mag_f4)"
        )
    x_scaled = _scaler_phot.transform(x)
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=_device)
    with torch.no_grad():
        logits = _model_phot(x_t).squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy().tolist()
    return PredictResponse(completeness=probs)


@app.get("/")
def root():
    return {
        "message": "Cluster Completeness API",
        "docs": "/docs",
        "health": "/health",
        "predict_phys": "POST /predict_phys",
        "predict_phot": "POST /predict_phot",
    }


def main():
    """CLI entry point: run uvicorn (host/port via env or defaults)."""
    import uvicorn

    host = os.environ.get("COMPLETENESS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("COMPLETENESS_API_PORT", "8000"))
    uvicorn.run("completeness_nn_api.serve:app", host=host, port=port)
