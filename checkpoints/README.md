# Checkpoints directory — put the four model files here

Place these four files in this directory (or any directory you prefer, then pass it to `deploy`):

| File | Description |
|------|-------------|
| `best_model_phys_model0.pt` | Phys model (mass, age, av → completeness) |
| `best_model_phot_model0.pt` | Phot model (5-band mag → completeness) |
| `scaler_phys_model0.pkl` | StandardScaler for phys (joblib) |
| `scaler_phot_model0.pkl` | StandardScaler for phot (joblib) |

If your training used a different `--outname`, use that name instead of `model0` and set `COMPLETENESS_OUTNAME` (or `deploy --outname yourname`).

**To ship weights with the package** so users only need `pip install`: put the four files in **`completeness_nn_api/checkpoints/`** instead, then build and publish to PyPI. See `docs/PUBLISH.md` §0.
