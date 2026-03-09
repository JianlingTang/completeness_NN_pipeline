# Publishing the package and API

Two ways to “publish”: (1) **Python package on PyPI** so anyone can `pip install cluster-completeness-pipeline`, and (2) **Host the API** so it’s reachable over the internet.

---

## 0. Ship NN weights with the package (recommended)

So that **users only need `pip install` and can run `predict()` with no extra download**:

1. Put the four checkpoint files in **`completeness_nn_api/checkpoints/`** (not the repo root `checkpoints/`):
   - `best_model_phys_model0.pt`
   - `best_model_phot_model0.pt`
   - `scaler_phys_model0.pkl`
   - `scaler_phot_model0.pkl`

2. Commit them to GitHub and publish the package to PyPI (see §1 below).  
   The build step includes `completeness_nn_api/checkpoints/*.pt` and `*.pkl` in the wheel, so they are installed with the package.

3. Users then:
   ```bash
   pip install cluster-completeness-pipeline
   ```
   ```python
   from completeness_nn_api import ngc628_completeness_predict as predict
   predict(phys=(mass, age, av))  # works immediately, no env or paths
   ```
   Or run the API: `deploy-completeness` (uses the bundled checkpoints by default).

So **GitHub hosts the repo (including the 4 files)**; **PyPI hosts the built wheel (including the 4 files)**; users get everything with one `pip install`.

---

## 1. Publish the Python package to PyPI

### Prerequisites

- PyPI account: [pypi.org/account/register](https://pypi.org/account/register/)
- For testing first: [test.pypi.org](https://test.pypi.org/)
- Optional: [Trusted Publisher](https://docs.pypi.org/trusted-publishers/) (GitHub Actions, no long‑lived token)

### One-time: add metadata (if not already in `pyproject.toml`)

Ensure `pyproject.toml` has at least:

- `name`, `version`, `description`, `readme`, `requires-python`
- `license` (e.g. `license = "MIT"`) or `license = { text = "MIT" }`
- `[project.urls]` (e.g. `Homepage`, `Repository`, `Documentation`)

See the [PyPI publishing guide](https://packaging.python.org/en/latest/tutorials/publishing-package-distribution-releases-using-github-actions/).

### Build and upload (manual)

From the repo root:

```bash
# Install build tools
pip install build twine

# Build wheel + sdist (creates dist/)
python -m build

# Check
twine check dist/*

# Test first (optional)
twine upload --repository testpypi dist/*
# Install from Test PyPI: pip install -i https://test.pypi.org/simple/ cluster-completeness-pipeline

# Publish to PyPI
twine upload dist/*
```

Use your PyPI API token when prompted (or set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=pypi-...`).

### Automated: GitHub Actions (Trusted Publisher)

1. On PyPI: **Account → Publishing → Add a new trusted publisher**  
   - Owner/repo, workflow name **`publish`** (file `.github/workflows/publish.yml`), environment (optional).
2. In the repo, the workflow **`.github/workflows/publish.yml`** is already added: it runs on **GitHub Release published**, runs `python -m build` and uploads via `pypa/gh-action-pypi-publish` (no token in secrets).

After that, creating a **GitHub Release** (e.g. tag `v0.1.0`) will build and publish to PyPI.

---

## 2. Publish the API (host it publicly)

The API needs the four checkpoint files and `COMPLETENESS_MODEL_DIR` / `COMPLETENESS_OUTNAME`. Two patterns:

- **A. Container (Docker)** — put checkpoints in the image or mount them; run anywhere (Cloud Run, ECS, Fly.io, Railway, etc.).
- **B. Platform config** — set env and (if needed) upload checkpoints to a volume or use a build step that downloads them.

### Option A: Docker

From repo root, with the four files in `checkpoints/`:

```bash
docker build -t cluster-completeness-api -f Dockerfile.api .
docker run -p 8000:8000 cluster-completeness-api
```

Then deploy the image to your preferred cloud (see below). The repo includes `Dockerfile.api` and `.dockerignore` for this.

### Option B: Platform-as-a-Service (no Docker)

- **Railway**  
  - New project → Deploy from GitHub.  
  - Build: `pip install -e ".[api]"` (or add `requirements.txt` with API deps).  
  - Start: `uvicorn completeness_nn_api.serve:app --host 0.0.0.0 --port $PORT`.  
  - Env: `COMPLETENESS_MODEL_DIR` = path to checkpoints (e.g. copy four files into repo under `checkpoints/` and set to `checkpoints`), `COMPLETENESS_OUTNAME=model0`.

- **Render**  
  - New Web Service → connect repo.  
  - Build: `pip install -e ".[api]"`. Start: `uvicorn completeness_nn_api.serve:app --host 0.0.0.0 --port $PORT`.  
  - Add env vars; if checkpoints are in repo, set `COMPLETENESS_MODEL_DIR=./checkpoints`.

- **Fly.io**  
  - `fly launch` in repo, then set env and (if needed) a volume for checkpoints, or bake them into the image with a Dockerfile.

- **Hugging Face Spaces** (with Docker)  
  - Space type: Docker. Use `Dockerfile.api`; ensure `PORT` is read (Spaces set `PORT`). Expose the same FastAPI app.

Important: set **`COMPLETENESS_MODEL_DIR`** (and optionally **`COMPLETENESS_OUTNAME`**) in the platform’s environment so the API finds the four files. If the four files are in the repo under `checkpoints/`, set `COMPLETENESS_MODEL_DIR=./checkpoints` (or the absolute path in the container).

---

## Summary

| Goal | Action |
|------|--------|
| **Package on PyPI** | `python -m build` → `twine upload dist/*` (or GitHub Actions Trusted Publisher). |
| **API on the internet** | Put four checkpoints in `checkpoints/` (or a dir you mount), then Docker (`Dockerfile.api`) or PaaS (Railway/Render/Fly.io) with `COMPLETENESS_MODEL_DIR` and start command `uvicorn completeness_nn_api.serve:app --host 0.0.0.0 --port $PORT`. |
