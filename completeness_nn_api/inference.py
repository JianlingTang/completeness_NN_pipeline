"""
Completeness NN inference: load saved model + scalers and run prediction.
Matches the MLP and scaling used in scripts/perform_ml_to_learn_completeness.py.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Same architecture as in perform_ml_to_learn_completeness.py."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model_and_scalers(
    model_dir: Path,
    outname: str,
    device: torch.device | None = None,
) -> tuple[MLP, MLP, object, object]:
    """
    Load phys model, phot model, scaler_phys, scaler_phot from a directory.

    Supports two layouts:

    1. Flat (single dir, four files): all in model_dir:
       - model_dir / f"best_model_phys_{outname}.pt"
       - model_dir / f"best_model_phot_{outname}.pt"
       - model_dir / f"scaler_phys_{outname}.pkl"
       - model_dir / f"scaler_phot_{outname}.pkl"

    2. Nested (sweep output): .pt in checkpoints/, scalers in model_dir:
       - model_dir / "checkpoints" / f"best_model_phys_{outname}.pt"
       - model_dir / "checkpoints" / f"best_model_phot_{outname}.pt"
       - model_dir / f"scaler_phys_{outname}.pkl"
       - model_dir / f"scaler_phot_{outname}.pkl"

    Returns
    -------
    (model_phys, model_phot, scaler_phys, scaler_phot)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(model_dir).resolve()
    flat_phys = model_dir / f"best_model_phys_{outname}.pt"
    nested_phys = model_dir / "checkpoints" / f"best_model_phys_{outname}.pt"

    if flat_phys.exists():
        ckpt_phys = flat_phys
        ckpt_phot = model_dir / f"best_model_phot_{outname}.pt"
    elif nested_phys.exists():
        ckpt_phys = nested_phys
        ckpt_phot = model_dir / "checkpoints" / f"best_model_phot_{outname}.pt"
    else:
        raise FileNotFoundError(
            f"Missing model artifacts: looked for {flat_phys} or {nested_phys}. "
            "Put the four files (best_model_phys_*, best_model_phot_*.pt, scaler_phys_*, scaler_phot_*.pkl) in one directory."
        )

    scaler_phys_path = model_dir / f"scaler_phys_{outname}.pkl"
    scaler_phot_path = model_dir / f"scaler_phot_{outname}.pkl"

    for p in (ckpt_phys, ckpt_phot, scaler_phys_path, scaler_phot_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    scaler_phys = joblib.load(scaler_phys_path)
    scaler_phot = joblib.load(scaler_phot_path)

    def _load_mlp(ckpt_path: Path) -> MLP:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["model_config"]
        model = MLP(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_hidden=cfg["n_hidden"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    model_phys = _load_mlp(ckpt_phys)
    model_phot = _load_mlp(ckpt_phot)
    return model_phys, model_phot, scaler_phys, scaler_phot
