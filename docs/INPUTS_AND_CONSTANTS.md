# Input files, constants, and magic numbers vs originals

This document compares the restructured pipeline with the original scripts and lists **required inputs** to run the full pipeline.

---

## 1. Input files (paths and patterns)

### Phase A – White-light select & inject

| Input | Original (`original_select_insert_white.py`) | Current (`generate_white_clusters.py` / pipeline) | Match? |
|-------|---------------------------------------------|---------------------------------------------------|--------|
| Galaxy list | `main_dir/galaxy_names.npy` | `main_dir/galaxy_names.npy` (when used) | ✓ |
| Filter dict | `fits_path/galaxy_filter_dict.npy` (for PHOTFLAM) | `main_dir/galaxy_filter_dict.npy` | ⚠ Different dir: original uses fits_path |
| Science FITS (flux scaling) | `fits_path/{galaxy_fullname}/*{filter_name}*.fits` (first match) | `fits_path/{galaxy_fullname}/*{filter_name}*drc.fits` (fallback `*sci.fits`) | ⚠ Current prefers drc then sci |
| White science frame | `gal_dir/white/` + `hlsp_legus_hst_*{gal}*_{filt}_*sci.fits` or `white_dualpop_s2n_white_remake.fits` | Same pattern (sci/drc in pipeline) | ✓ |
| Readme | `gal_dir/automatic_catalog*_{gal}.readme` | Same | ✓ |
| header_info | `gal_dir/header_info_{gal}.txt` | Same (also `header_info_{gal_short}.txt`) | ✓ |
| Aperture correction | `gal_dir/avg_aperture_correction_{gal}.txt` | Same | ✓ |
| SLUG library | `libdir/tang_padova*_cluster_phot.fits` (first file only), slugpy `read_cluster` | `libdir/flat_in_logm_cluster_phot.fits` or `flat_in_logm` (cluster_pipeline slug_reader) | ⚠ Different library file/name |
| PSF | `PSFpath/psf_*_{cam}_{filt}.fits` | Same pattern | ✓ |
| BAOlab | External binary path | Same | ✓ |

### 5-filter inject

| Input | Original (`original_inject_to_5filters.py`) | Current (`inject_clusters_to_5filters.py`) | Match? |
|-------|--------------------------------------------|--------------------------------------------|--------|
| Galaxy list | `main_dir/galaxy_names.npy` | Same | ✓ |
| Filter dict | `main_dir/galaxy_filter_dict.npy` | Same | ✓ |
| Readme | `gal_dir/automatic_catalog*_{gal}.readme` | Same; two aperture patterns, distance + CI required when readme exists | ✓ |
| header_info | `gal_dir/header_info_{gal}.txt` | Resolved via `_resolve_gal_data_dir`; `header_info_{gal}.txt` or `header_info_{gal_short}.txt` | ✓ |
| Science FITS | `*sci.fits` or `*drc.fits` by galaxy | Same (sci/drc) | ✓ |
| Aperture correction | `gal_dir/avg_aperture_correction_{gal}.txt` | Same (path from gal_data_dir) | ✓ |
| Physprop (mag_BAO, etc.) | From Phase A output | Same: `main_dir/physprop/mag_BAO_select_*_frame*_reff*_{outname}.npy` etc. | ✓ |

### Photometry + CI + catalogue

| Input | Original (`original_photometry_on_5_and_CI.py`) | Current (pipeline + `ci_filter.py` + `catalogue_filters.py`) | Match? |
|-------|-------------------------------------------------|-------------------------------------------------------------|--------|
| Filter dict | `galaxy_filter_dict.npy` (cwd) | `main_dir/galaxy_filter_dict.npy` | ✓ (path from config) |
| Readme | `galdir/automatic_catalog*_{gal}.readme` | `gal_dir/automatic_catalog*_{gal_short}.readme` | ✓ |
| header_info | `galdir/header_info_{gal}.txt` | `gal_dir/header_info_{gal_short}.txt` (or galaxy_id) | ✓ |
| Aperture correction | `galdir/avg_aperture_correction_{gal}.txt` (required) | `main_dir/{galaxy_id}/` or `fits_path/{galaxy_id}/`; `avg_aperture_correction_{galaxy_id}.txt` or `_{gal_short}.txt`; **required** (raises if missing) | ✓ |
| Matched coords | `matched_coords/matched_frame{frame}_{outname}_reff{reff}.txt` | Same pattern | ✓ |
| Science FITS (per filter) | `galdir/hlsp_legus_hst_*{gal}_{filt}_*sci.fits` | Pipeline uses metadata FITS paths (drc or sci) | ✓ |

---

## 2. Constants and magic numbers

| Constant | Original | Current | Match? |
|----------|----------|---------|--------|
| **White / injection** | | | |
| sigma_pc (Gaussian white footprint) | 100 | 100 | ✓ |
| minsep | False | False | ✓ |
| min_separation | 3 × (reff in px from arctan(reff*3*pc / galdist)) | Same | ✓ |
| White scale factors (dual pop) | [55.8,45.3,44.2,65.7,29.3] and [0.5,0.8,3.1,11.2,13.7]; 0.5*(img21ms+img24rgb) | Same | ✓ |
| maglim (BAOlab test frames) | `[mag_BAO.min(), mag_BAO.max()]` (from radius-masked library) | **Hardcoded [-4, 4]** | ❌ Fixed below |
| **5-filter inject** | | | |
| sigma_pc | 100 | 100 | ✓ |
| baozpoint | 1e10 | 1e10 | ✓ |
| maglim | [10, 26] | [10, 26] | ✓ |
| tolerance (matching) | 3 | 3 | ✓ |
| **Photometry** | | | |
| merr_cut | 0.3 | 0.3 | ✓ |
| Science aperture (user) | From readme (e.g. 4 px NGC 628c) | From metadata / readme; fallback 4 for ngc628-c | ✓ |
| Sky annulus | inner 7 px, width 1 px | 7 px, 1 px | ✓ |
| Aperture radii in .mag | 1, 3, user (e.g. 4) | Same | ✓ |
| **CI cut** | | | |
| CI formula | mag(1px) - mag(3px) | Same | ✓ |
| CI threshold | From readme (e.g. 1.4) | From metadata; default 1.4 | ✓ |
| Aperture correction | Applied to mag_4, merr_4 from avg_aperture_correction_*.txt | Same | ✓ |
| **Catalogue** | | | |
| M_V cut | — | M_V ≤ -6 (LEGUS) | N/A (new in pipeline) |
| Four-band merr | — | ≥4 bands with merr ≤ 0.3 | ✓ |
| **Phase A library filters** | | | |
| Bright limit (absolute) | — | M ≥ 15 mag in all filters (current) | Current addition |
| Faint limit (apparent) | — | All bands > 18 mag (current) | Current addition |
| Original | No such cut in select_insert_white | — | — |

---

## 3. Required inputs (concise)

### Per run (project level)

- **main_dir**: Project root (contains `galaxy_names.npy`, `galaxy_filter_dict.npy`, and per-galaxy dirs).
- **fits_path**: Directory containing per-galaxy data (can equal main_dir). Must have `{galaxy_id}/` with:
  - Science FITS per filter (`*{filter}*drc.fits` or `*sci.fits`).
  - `header_info_{gal_short}.txt` (or `header_info_{galaxy_id}.txt`).
  - `avg_aperture_correction_{galaxy_id}.txt` (or `avg_aperture_correction_{gal_short}.txt`) for photometry.
  - Optionally `automatic_catalog*_{gal_short}.readme` (aperture, dmod, CI).
- **PSF path**: Directory with `psf_*_{cam}_{filt}.fits`.
- **BAOlab**: Path to BAOlab binary (Phase A white injection).
- **SLUG library** (Phase A): e.g. `COMP_SLUG_LIB_DIR` or `main_dir/SLUG_library`; current code uses `flat_in_logm` (or `flat_in_logm_cluster_phot.fits`). Original uses `tang_padova*_cluster_phot.fits`.

### Per galaxy

- `main_dir/galaxy_filter_dict.npy` (or under fits_path in original Phase A) with entry for galaxy short name (e.g. `ngc628-c` → `ngc628-c`).
- `main_dir/{galaxy_id}/` (or `fits_path/{galaxy_id}/`): FITS, header_info, avg_aperture_correction, optionally readme.
- For **photometry + catalogue**: `avg_aperture_correction_*.txt` is **required** (pipeline raises if missing).

### Optional / behaviour-changing

- **input_coords**: If set, Phase A uses pre-defined coordinates instead of SLUG sampling.
- **dmod**: Distance modulus (from readme or config); needed for M_V cut and apparent-mag limits.
- **Readme**: If present, aperture (two patterns in inject script), distance modulus, and CI are read; inject script requires all when readme exists.

---

## 4. Fix applied: maglim in Phase A

In the original `original_select_insert_white.py`, **maglim** for BAOlab test frames is set from the actual library-derived magnitude range: `maglim = [mag_BAO.min(), mag_BAO.max()]` (after radius mask). In `generate_white_clusters.py` it was hardcoded to **[-4, 4]**. A fix is applied so that when Phase A has run and saved `mag_BAO_select_*_frame{i_frame}_reff{eradius}_{outname}.npy`, `generate_white()` loads that file and sets `maglim = [arr.min(), arr.max()]`; otherwise it keeps the default `[-4, 4]` (e.g. for input_coords or missing physprop).
