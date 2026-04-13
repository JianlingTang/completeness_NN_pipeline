# Pipeline: Files Used by Stage

This document lists the **files and directories** the pipeline reads or writes for each stage. Paths are under `main_dir` (project root or `COMP_MAIN_DIR`) unless noted.

---

## Stage overview

| Stage | Name        | Script / module |
|-------|-------------|------------------|
| 0 (Phase A) | White injection | `scripts/generate_white_clusters.py` |
| 1 | Injection (copy frame+coords) | `pipeline_runner.py` |
| 2 | Detection | SExtractor via `sextractor_runner.py` |
| 3 | Matching | `coordinate_matcher.py`, `pipeline_runner.py` |
| 4 | Photometry | `aperture_photometry.py`, `ci_filter.py`, `inject_clusters_to_5filters.py` |
| 5 | Catalogue | `catalogue_filters.py`, `label_builder.py` |
| 6 | Dataset / ML | `build_ml_inputs.py`, `dataset_builder.py` |

---

## Phase A: White-light injection (`scripts/generate_white_clusters.py`)

**Inputs**

| File / directory | Purpose |
|------------------|--------|
| `galaxy_filter_dict.npy` | Galaxy → (filters, instruments); used for filter order and PSF/SLUG lookup. |
| `galaxy_names.npy` | Optional; list of galaxy IDs for batch. |
| `{galaxy}/ngc628-c_white.fits` or `--sciframe PATH` | White-light science frame. |
| `{galaxy}/r2_wl_aa_{gal}.config` | SExtractor config for white (path can come from gal dir). |
| `{galaxy}/automatic_catalog*_{gal}.readme` | Aperture radius, distance modulus, CI cut (regex-parsed). |
| `{galaxy}/header_info_{gal}.txt` | Zeropoints per filter (optional). |
| `SLUG_library/flat_in_logm_cluster_phot.fits` | SLUG cluster library (and optionally `*_cluster_phot.fits`, `output_lib/*.fits`). |
| `PSF_files/` or `COMP_PSF_PATH` | PSF FITS (e.g. `psf_*_{cam}_{filt}.fits`). |
| BAOlab binary | From `COMP_BAO_PATH` or project `.deps/local/bin`. |
| `{fits_path}/{galaxy}/*{filter}*drc.fits` | Per-filter science FITS for metadata/headers. |

**Outputs**

| File / directory | Purpose |
|------------------|--------|
| `{galaxy}/white/synthetic_fits/` | `*_frame{i}_{outname}_reff{r}.fits` synthetic white frames. |
| `{galaxy}/white/white_position_{i}_{outname}_reff{r}.txt` | Injected coords (x y mag), x=col y=row, same as legus_original_pipeline. |
| `physprop/mass_select_model{mrmodel}_frame{i}_reff{r}_{outname}.npy` | Selected mass. |
| `physprop/age_select_model{mrmodel}_frame{i}_reff{r}_{outname}.npy` | Selected age. |
| `physprop/av_select_model{mrmodel}_frame{i}_reff{r}_{outname}.npy` | Selected A_V. |
| `physprop/mag_BAO_select_...npy`, `physprop/mag_VEGA_select_...npy` | Selected magnitudes (BAO/VEGA). |

---

## Stage 1: Injection (copy to temp)

**Inputs**

| File / directory | Purpose |
|------------------|--------|
| `{galaxy}/white/synthetic_fits/*_frame{i}_{outname}_reff{r}.fits` | One synthetic white frame per (frame_id, reff). |
| `{galaxy}/white/white_position_{i}_{outname}_reff{r}.txt` | Injected coords for that frame/reff. |

**Outputs** (in temp dir, then discarded unless `keep_frames`)

| File | Purpose |
|------|--------|
| `injected.fits` | Copy of synthetic frame. |
| `injected_coords.txt` | Copy of coords. |

---

## Stage 2: Detection (SExtractor)

**Inputs**

| File / directory | Purpose |
|------------------|--------|
| `injected.fits` | Frame to run SExtractor on (in temp dir). |
| `output.param` | SExtractor PARAMETERS_NAME (X_IMAGE, Y_IMAGE, etc.). |
| `default.nnw` | SExtractor STARNNW_NAME. |
| `{galaxy}/r2_wl_aa_{gal}.config` or `sextractor_config_path` | SExtractor config (-c). |

**Outputs** (temp dir)

| File | Purpose |
|------|--------|
| `det_frame{i}_reff{r}.cat` | SExtractor catalog. |
| `injected.coo` | Two-column x y derived from catalog for matching. |

---

## Stage 3: Matching

**Inputs**

| File | Purpose |
|------|--------|
| `injected_coords.txt` | Injected (y,x,mag) → converted to (x,y). |
| `injected.coo` | Detected positions from SExtractor. |

**Outputs**

| File / directory | Purpose |
|------------------|--------|
| `{galaxy}/white/matched_coords/matched_frame{i}_{outname}_reff{r}.txt` | Matched coords (and optional `_cluster_ids.txt`). |
| `{galaxy}/white/diagnostics/match_summary_frame{i}_reff{r}_{outname}.txt` | Match summary. |
| `{galaxy}/white/catalogue/match_results_frame{i}_{outname}_reff{r}.parquet` | Per-cluster match 0/1 + coords. |
| `{galaxy}/white/detection_labels/detection_labels_white_match_frame{i}_{outname}_reff{r}.npy` | Binary labels (white-match only). |

---

## Stage 4: Photometry (optional, 5-filter)

**Inputs**

| File / directory | Purpose |
|------------------|--------|
| `{galaxy}/white/matched_coords/matched_frame{i}_{outname}_reff{r}.txt` | Matched positions for photometry. |
| `matched_..._cluster_ids.txt` | Cluster IDs for matched rows. |
| `physprop/mag_VEGA_select_model{mrmodel}_frame{i}_reff{r}_{outname}.npy` | Vega mags for injection onto 5-filter images. |
| `{galaxy}/{filter}/synthetic_fits/*_frame{i}_{outname}_reff{r}.fits` | Per-filter synthetic frame (written by inject script). |
| `galaxy_filter_dict.npy` | Filter list. |
| `{galaxy}/automatic_catalog*_{gal}.readme` | Aperture, CI cut. |
| `{galaxy}/header_info_{gal}.txt` | Zeropoints. |
| `scripts/inject_clusters_to_5filters.py` | Called to create per-filter synthetic FITS from matched coords. |

**Outputs**

| File / directory | Purpose |
|------------------|--------|
| `{galaxy}/{filter}/photometry/*.mag`, `*.txt` | IRAF aperture photometry output. |
| `{galaxy}/white/catalogue/photometry_frame{i}_{outname}_reff{r}.parquet` | Photometry table (mag, merr, ci, passes_ci, etc.). |
| `{galaxy}/white/catalogue/catalogue_frame{i}_{outname}_reff{r}.parquet` | After CI/merr cuts; `in_catalogue` column. |

---

## Stage 5: Catalogue (CI/merr → final labels)

**Inputs**

| File | Purpose |
|------|--------|
| `photometry_frame{i}_{outname}_reff{r}.parquet` | From stage 4. |
| `match_results_frame{i}_{outname}_reff{r}.parquet` | From stage 3. |

**Outputs**

| File | Purpose |
|------|--------|
| `{galaxy}/white/catalogue/catalogue_frame{i}_{outname}_reff{r}.parquet` | With `in_catalogue`. |
| `{galaxy}/white/detection_labels/detection_frame{i}_{outname}_reff{r}.npy` | Final binary labels (after CI). |

---

## Stage 6: Dataset / ML inputs

**Inputs** (for `build_ml_inputs.py`)

| File / directory | Purpose |
|------------------|--------|
| `{galaxy}/white/detection_labels/detection_frame{i}_{outname}_reff{r}.npy` | Or `detection_labels_white_match_...` if `--use-white-match`. |
| `physprop/mass_select_model{mrmodel}_frame{i}_reff{int(r)}_{outname}.npy` | Mass per cluster. |
| `physprop/age_select_...npy`, `physprop/av_select_...npy` | Age, A_V. |
| `physprop/mag_VEGA_select_...npy` | Vega mags. |

**Outputs**

| File | Purpose |
|------|--------|
| `det_3d.npy` | Shape (n_clusters, n_frames, n_reff), 0/1. |
| `allprop.npz` | mass, age, av, phot (CFR flatten order). |

For **dataset_builder** (parquet/npy dataset):

**Inputs**

| File | Purpose |
|------|--------|
| Injected parquet | cluster_id, mass, age, av, mag_f0..4. |
| Match parquet | cluster_id, matched, ... |
| Catalogue parquet | cluster_id, in_catalogue. |
| Photometry parquet (optional) | For magnitudes. |

**Outputs**

| File | Purpose |
|------|--------|
| `{prefix}_cluster_properties.npy` | mass, age, av. |
| `{prefix}_magnitudes.npy` | mag_f0..4. |
| `{prefix}_detection_labels.npy` | Binary labels. |
| Dataset parquet | Full table. |

---

## Summary: minimal file set to run all stages

**Config / metadata (required)**

- `galaxy_filter_dict.npy`
- `galaxy_names.npy` (optional)
- `{galaxy}/ngc628-c_white.fits` (or `--sciframe`)
- `{galaxy}/r2_wl_aa_{gal}.config`
- `output.param`
- `default.nnw`
- `{galaxy}/automatic_catalog*_{gal}.readme`
- `{galaxy}/header_info_{gal}.txt` (optional but recommended for photometry)

**Data**

- `SLUG_library/` (at least `flat_in_logm_cluster_phot.fits`)
- `PSF_files/` or `COMP_PSF_PATH`
- BAOlab binary (`COMP_BAO_PATH`)

**For stages 4–5 (photometry + catalogue)**

- Per-filter science FITS under `{galaxy}/{filter}/` (e.g. `*F275W*drc.fits`)
- `scripts/inject_clusters_to_5filters.py` (for 5-filter injection)

**Environment (optional)**

- `COMP_MAIN_DIR`, `COMP_FITS_PATH`, `COMP_PSF_PATH`, `COMP_BAO_PATH`, `COMP_SLUG_LIB_DIR`, `COMP_OUTPUT_LIB_DIR`

See **`docs/RUNNING.md`** for run commands and **`cluster_pipeline/config/pipeline_config.py`** for path helpers.
