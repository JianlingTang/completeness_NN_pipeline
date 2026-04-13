# Catalogue inclusion criteria (in_catalogue)

Source: LEGUS-style selection (2015LEGUS, Adamo et al.).  
Selection is **Stage A** (automatic) then **Stage B** (cluster catalogue).  
A cluster is **in_catalogue** only if it passes **Stage A** (CI + Stage A merr) and **Stage B** (Stage B merr + M_V).

---

## Stage A (automatic)

### 1. CI (Concentration Index) — V-band only

- Set during photometry; stored as `passes_ci`; catalogue uses max over filters per cluster.
- **Rule:** **Keep** if CI ≥ threshold (e.g. 1.4) on V-band (F555W); **discard** if CI < threshold (point-like = star).
- **Code:** `cluster_pipeline/photometry/ci_filter.py` — `apply_ci_cut(..., ci_threshold, filter_is_vband=True for F555W)`.

### 2. Stage A merr

- **Rule:** **Keep** only if **V-band merr ≤ 0.3** and **at least one of B or I merr ≤ 0.3** (B = F435W, I = F814W).  
  **Discard** if V_merr > 0.3 or (B_merr > 0.3 and I_merr > 0.3).
- **Code:** `passes_stage1_merr` in `catalogue_filters.py`: `v_ok and (b_ok or i_ok)`.

**Check:** V_merr ≤ 0.3 and (B_merr ≤ 0.3 or I_merr ≤ 0.3).

---

## Stage B (cluster catalogue; applied to clusters passing Stage A)

### 3. Stage B merr

- **Rule:** **Keep** only if **at least 4 bands** have merr ≤ 0.3 (i.e. at most 1 filter with merr > 0.3).
- **Code:** `passes_stage2_merr` in `catalogue_filters.py`: `n_bad <= 1`.

**Check:** Number of filters with merr > 0.3 must be ≤ 1.

### 4. M_V (absolute V magnitude)

- **Rule:** **Keep** if M_V ≤ −6; **discard** if M_V > −6. M_V = m_V − dmod (apparent V from photometry).
- **Code:** `passes_MV` in `catalogue_filters.py`; if dmod is None, `passes_MV = 1`.

**Check:** M_V = m_V − dmod; require M_V ≤ −6 (turnover at apparent V ≈ dmod − 6 mag).

---

## Combined

```text
in_catalogue = (passes_ci == 1)
             & (passes_stage1_merr == 1)   # Stage A merr
             & (passes_stage2_merr == 1)   # Stage B merr (>=4 bands)
             & (passes_MV == 1)            # Stage B M_V <= -6
```

All four must be 1.

---

## Defaults summary (for校对)

| Criterion        | Default / value | Where set |
|------------------|------------------|-----------|
| CI threshold     | 1.4              | Galaxy readme or `ci_cut`; else 1.4 in `pipeline_runner.py` |
| merr_cut         | 0.3 mag          | `catalogue_filters.MERR_CUT`, `config.merr_cut` |
| V band           | F555W            | Stage A: V merr≤0.3; Stage B: M_V≤−6 |
| B band           | F435W            | Stage A: B or I merr≤0.3 |
| I band           | F814W            | Stage A: B or I merr≤0.3 |
| M_V cut          | M_V ≤ −6         | Stage B; dmod from metadata or config (e.g. 29.98) |

---

## Code references

- **Catalogue filters:** `cluster_pipeline/catalogue/catalogue_filters.py` — `apply_catalogue_filters()`, `in_catalogue` assignment.
- **CI and passes_ci:** `cluster_pipeline/photometry/ci_filter.py` — `apply_ci_cut()`; pipeline writes `passes_ci` in `cluster_pipeline/pipeline/pipeline_runner.py` (photometry + catalogue step).
- **Config:** `cluster_pipeline/config/pipeline_config.py` — `merr_cut`, `dmod`; `cluster_pipeline/data/galaxy_metadata.py` — `ci_cut`, `distance_modulus`.
