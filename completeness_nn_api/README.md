# Completeness Neural Network API

Deploy your trained completeness model (from `perform_ml_to_learn_completeness.py` hyperparameter sweep) as an HTTP API. You can **skip training** by placing the four checkpoint files in a directory (e.g. repo `checkpoints/`) and running the deploy command.

## One-step deploy (recommended)

1. Put the four files in one directory (e.g. `checkpoints/` in the repo):

   | File | Description |
   |------|-------------|
   | `best_model_phys_model0.pt` | Best phys model (mass, age, av → completeness) |
   | `best_model_phot_model0.pt` | Best phot model (5-band mag → completeness) |
   | `scaler_phys_model0.pkl` | StandardScaler for phys (joblib) |
   | `scaler_phot_model0.pkl` | StandardScaler for phot (joblib) |

2. From repo root:

   ```bash
   pip install -e ".[api]"
   deploy-completeness
   ```

   Or with a custom directory and install in one go:

   ```bash
   python scripts/deploy.py --model-dir /path/to/checkpoints --install
   ```

   - **Docs (Swagger):** http://localhost:8000/docs  
   - **Health:** http://localhost:8000/health  

Both **flat** (all four files in one dir) and **nested** (`checkpoints/*.pt` + scalers in parent) layouts are supported. Set `COMPLETENESS_OUTNAME` if your files use a different suffix (e.g. `model1`).

## Programmatic use (no HTTP)

```python
from completeness_nn_api import ngc628_completeness_predict as predict
p = predict(phys=(mass, age, av))   # or predict(phot=mag_5band)
```

## Legacy layout

Nested layout (sweep output): `model_dir/checkpoints/*.pt` and `model_dir/scaler_*.pkl`. Set `COMPLETENESS_MODEL_DIR` and `COMPLETENESS_OUTNAME` and run `serve-completeness-api` or `uvicorn completeness_nn_api.serve:app`.

## Endpoints

### POST /predict_phys

Predict completeness from physical properties. Body (JSON):

```json
{
  "mass": [1e4, 2e4, 3e4],
  "age": [1e8, 2e8, 3e8],
  "av": [0.1, 0.2, 0.3]
}
```

Units: mass [M_sun], age [yr], av [mag]. Returns:

```json
{ "completeness": [0.85, 0.72, 0.61] }
```

### POST /predict_phot

Predict completeness from 5-band magnitudes. Body (JSON):

```json
{
  "phot": [
    [18.0, 18.1, 18.2, 18.3, 18.4],
    [19.0, 19.1, 19.2, 19.3, 19.4]
  ]
}
```

Order: mag_f0..mag_f4 (same as training, e.g. F275W, F336W, F438W, F555W, F814W). Returns:

```json
{ "completeness": [0.9, 0.5] }
```

## Example: curl

```bash
curl -X POST http://localhost:8000/predict_phys \
  -H "Content-Type: application/json" \
  -d '{"mass":[1e4],"age":[1e8],"av":[0.1]}'
```

## Deploying externally

- Run behind a reverse proxy (nginx, Caddy) with HTTPS.
- Use a process manager (systemd, supervisord) or a host like Railway, Render, Fly.io, and set `COMPLETENESS_MODEL_DIR` to where you upload the `.pt` and `.pkl` files (e.g. a volume or build step that downloads from S3/GCS).
