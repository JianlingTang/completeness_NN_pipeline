# Files to Include for Git (Pipeline-Only Push)

Commit only these paths so the repo contains the pipeline and ML code, no data or heavy outputs.

## Root

- `README.md`
- `pyproject.toml`
- `pytest.ini`
- `requirements.txt`
- `.gitignore`
- `generate_white_clusters.py`
- `perform_photometry_ci_cut_on_5filters.py`
- `perform_ml_to_learn_completeness.py`
- `nn_utils.py`

## Scripts

- `scripts/run_small_test.py`
- `scripts/inject_clusters_to_5filters.py`
- `scripts/build_ml_inputs.py`
- `scripts/plot_completeness_mag_mass_age.py`

## Cluster pipeline package

- `cluster_pipeline/__init__.py`
- `cluster_pipeline/config/__init__.py`
- `cluster_pipeline/config/pipeline_config.py`
- `cluster_pipeline/data/__init__.py`
- `cluster_pipeline/data/models.py`
- `cluster_pipeline/data/schemas.py`
- `cluster_pipeline/data/galaxy_metadata.py`
- `cluster_pipeline/data/slug_reader.py`
- `cluster_pipeline/data/cluster_library.py`
- `cluster_pipeline/data/slug_library_loader.py`
- `cluster_pipeline/detection/__init__.py`
- `cluster_pipeline/detection/sextractor_runner.py`
- `cluster_pipeline/matching/__init__.py`
- `cluster_pipeline/matching/coordinate_matcher.py`
- `cluster_pipeline/pipeline/__init__.py`
- `cluster_pipeline/pipeline/pipeline_runner.py`
- `cluster_pipeline/pipeline/stages.py`
- `cluster_pipeline/pipeline/injection_5filter.py`
- `cluster_pipeline/pipeline/diagnostics.py`
- `cluster_pipeline/pipeline/ast_pipeline.py`
- `cluster_pipeline/pipeline/manifest.py`
- `cluster_pipeline/photometry/__init__.py`
- `cluster_pipeline/photometry/aperture_photometry.py`
- `cluster_pipeline/photometry/ci_filter.py`
- `cluster_pipeline/catalogue/__init__.py`
- `cluster_pipeline/catalogue/catalogue_filters.py`
- `cluster_pipeline/catalogue/label_builder.py`
- `cluster_pipeline/utils/__init__.py`
- `cluster_pipeline/utils/filesystem.py`
- `cluster_pipeline/utils/logging_utils.py`
- `cluster_pipeline/utils/mag_parser.py`
- `cluster_pipeline/utils/fits_arithmetic.py`
- `cluster_pipeline/dataset/__init__.py`
- `cluster_pipeline/dataset/dataset_builder.py`
- `cluster_pipeline/injection/__init__.py`
- `cluster_pipeline/simulation/__init__.py`

## Docs

- `docs/RUNNING.md` – How to run: required files, commands, env vars
- `docs/DEPLOY_FOR_PAPER.md`
- `docs/FILES_FOR_GIT.md`

## CI

- `.github/workflows/ci.yml`

## Tests

- `tests/__init__.py`
- `tests/conftest.py`
- `tests/README.md`
- `tests/unit/__init__.py`
- `tests/unit/test_diagnostics.py`
- `tests/unit/test_ci_filter.py`
- `tests/unit/test_coordinate_matcher.py`
- `tests/unit/test_label_builder.py`
- `tests/integration/__init__.py`
- `tests/integration/test_completeness_visual.py`
- `tests/integration/test_build_ml_inputs.py`
- `tests/e2e/__init__.py`
- `tests/e2e/test_pipeline_smoke.py`

## Do not commit

- `.venv/`, `.deps/`, `__pycache__/`
- `*.npy`, `*.npz`, `*.fits`, `*.parquet`
- `SLUG_library/`, `PSF_files/`, galaxy data dirs
- Pipeline outputs: `physprop/`, `ngc628-c/white/detection_labels/`, `synthetic_fits/`, etc.
- Notebooks (unless you add them explicitly), `.pytest_cache/`

Use the shell commands in the README to add only the paths above and push.
