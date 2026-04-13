# Post-refactor audit report

**Date:** 2025-03-04  
**Scope:** cluster-completeness-pipeline (astrophysics research code).  
**Constraints:** Correctness, backwards compatibility, reproducibility; minimal-risk cleanup only.

---

## 1. Audit summary

- **Entry points:** `scripts/run_pipeline.py` (main), `scripts/run_stage123_and_plot_diagnostics.py`, `scripts/generate_white_clusters.py`, `scripts/build_ml_inputs.py`, `scripts/perform_ml_to_learn_completeness.py`, `scripts/plot_completeness_mag_mass_age.py`, `scripts/inject_clusters_to_5filters.py`, `scripts/perform_photometry_ci_cut_on_5filters.py` (standalone reference), `scripts/extract_white.py`, `scripts/sample_slug_white_mag.py`, `scripts/setup_env.sh`, `scripts/generate_x11_stubs.py`.
- **Pipeline package:** `cluster_pipeline/` — config, data (models, schemas, slug_reader, galaxy_metadata, cluster_library, slug_library_loader), detection (SExtractor), matching (coordinate_matcher), pipeline (pipeline_runner, ast_pipeline, stages, injection_5filter, diagnostics, manifest), photometry (aperture_photometry, ci_filter), catalogue (catalogue_filters, label_builder), dataset (dataset_builder), utils (filesystem, fits_arithmetic, logging_utils, mag_parser).
- **Tests:** Root-level `tests/test_*.py` and `tests/unit/`, `tests/integration/`, `tests/e2e/` with some overlapping coverage (e.g. coordinate_matcher, label_builder) but different style (root = class-based + fixtures, unit = function-based). Both kept; no duplication removed to avoid changing behaviour.
- **Issue found and fixed:** Root-level tests referenced fixtures (`sample_injected_coords`, `sample_detected_coords`, `sample_coords_three_col`, `sample_cluster_ids`, `sample_photometry_parquet`, `sample_injected_parquet`, `sample_match_parquet`, `sample_catalogue_parquet`) that were not defined in `conftest.py`, causing **10+ test collection/setup errors**. These fixtures were added to `tests/conftest.py` so all root-level tests run.

---

## 2. Duplicated code found

- **Tests:** `tests/test_coordinate_matcher.py` and `tests/unit/test_coordinate_matcher.py` both test `match_coordinates`, `load_coords`, and `CoordinateMatcher`; `tests/test_label_builder.py` and `tests/unit/test_label_builder.py` both test `build_final_detection` and `save_final_detection`. Intentionally not consolidated: root tests use shared fixtures and class-based layout; unit tests are smaller and more isolated. Both layers are retained.
- **Pipeline entry points:** `run_galaxy_pipeline` (pipeline_runner) is the main path used by `run_pipeline.py`; `run_ast_pipeline` (ast_pipeline) is exported and may be used by HPC or external workflows. No consolidation or removal.
- **`load_coords` vs `load_coords_white_position`:** Both in `coordinate_matcher.py`; same file format (x y [mag]). `load_coords_white_position` is for white_position files from generate_white_clusters; both return (x, y) = (col, row). No duplication to remove.
- **`slug_library_loader.py`:** Re-exports `load_slug_library` from `cluster_library.py`. No other code imports `slug_library_loader`; only `cluster_library.load_slug_library` exists. Kept as a documented alias for refactor/spec compatibility; not removed.

---

## 3. Unused / dead code removed

- **None.** No code was deleted. Conservative rule: only remove when clearly unreferenced and safe.  
- **`ci.py` (root):** Not referenced by any script or doc as an entry point; it imports `cluster_pipeline.data.schemas.get_required_columns`. Not removed; marked as uncertain (see §7).  
- **Commented-out blocks:** No large commented-out blocks were removed to avoid affecting reproducibility or future reference.

---

## 4. Docstrings added/improved

- **`tests/conftest.py`:**  
  - Added docstrings for `project_root` and for all new fixtures: `sample_injected_coords`, `sample_detected_coords`, `sample_coords_three_col`, `sample_cluster_ids`, `sample_photometry_parquet`, `sample_injected_parquet`, `sample_match_parquet`, `sample_catalogue_parquet`.  
  - Each describes purpose and, where relevant, content (e.g. column layout, number of rows).
- **Existing docstrings:** Core pipeline modules already had usable docstrings (e.g. `coordinate_matcher.py`, `ci_filter.py`, `catalogue_filters.py`, `dataset_builder.py`, `schemas.py`, `fits_arithmetic.py`, `filesystem.py`, `mag_parser.py`). No speculative changes made.

---

## 5. Tests run

- **Command:** `pytest tests/` (from repo root).  
- **Scope:** All tests under `tests/` (unit, integration, e2e, and root-level `test_*.py`).

---

## 6. Pytest results

```
======================== 111 passed, 1 skipped in 0.98s ========================
```

- **Passed:** 111.  
- **Skipped:** 1 (e.g. `test_run_pipeline_plot_only` or similar conditional/slow test).  
- **Failed:** 0.  
- **Errors:** 0 (after adding the missing fixtures to `conftest.py`).

**Pre-existing vs introduced:** The errors that appeared at the start of the audit were **pre-existing**: missing fixtures in `conftest.py` for root-level tests. The **only change** made to fix tests was adding those fixtures; **no test logic or scientific behaviour was changed**. All current passes are after that single, conservative fix.

---

## 7. Remaining uncertain areas (not deleted)

- **`ci.py` (root):** Standalone script; not listed in `scripts/README.md` or `docs/SCRIPTS.md`. Imports `cluster_pipeline.data.schemas.get_required_columns`. Could be legacy or used by hand/HPC. **Recommendation:** Keep; if you confirm it is unused, add a one-line comment or move to a `legacy/` or `archive/` and document.
- **`run_ast_pipeline`:** Exported from `cluster_pipeline.pipeline`; no in-repo callers found. May be used by external or HPC scripts. **Recommendation:** Keep.
- **`slug_library_loader`:** No in-repo imports; thin wrapper around `cluster_library.load_slug_library`. **Recommendation:** Keep for naming/spec compatibility.
- **Root-level vs unit tests:** Overlap in coverage for coordinate_matcher and label_builder. **Recommendation:** Keep both; document in `tests/README.md` that root tests use shared fixtures and class-based layout, unit tests are more isolated.

---

## 8. File and behaviour contracts

- No changes to: FITS I/O, parquet schemas, catalogue columns, file naming, CLI arguments, or pipeline stage order.  
- No `os.chdir`-dependent logic introduced.  
- Multiprocessing usage and random-seed semantics were not modified.  
- Completeness calculations, photometry formulae, CI threshold, and selection logic were not altered.

---

## 9. Summary of code edits

| File | Change |
|------|--------|
| `tests/conftest.py` | Added fixtures: `sample_injected_coords`, `sample_detected_coords`, `sample_coords_three_col`, `sample_cluster_ids`, `sample_photometry_parquet`, `sample_injected_parquet`, `sample_match_parquet`, `sample_catalogue_parquet`; added docstrings for `project_root` and all new fixtures. |

No other files were modified. No dead code was removed; no scientific or pipeline logic was changed.
