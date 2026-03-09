# Completeness workflow: scripts, how to run, and assumptions

## 0. Photometry / CI cut logic (current implementation)

- **Not**: running 5-filter photometry on the white synthetic image, or copying the same white image to five filters.
- **Is**: use white-light detection + matching to get **matched_coords** (same set of clusters, same coordinates); **inject** those synthetic sources **onto the real HLSP F*W drc/sci science images** (one science image per filter) to get "science + injected" FITS; then run photometry and CI cut on those images using **matched_coords**.
- The same cluster has **the same coordinates** on white and on all 5 filters (the matched positions); magnitudes per filter come from physprop `mag_VEGA_select` (one column per filter).

In practice: when the pipeline runs with photometry and `inject_5filter_script` is set, it writes per-filter `(x, y, mag)` files from matched_coords + physprop, then calls **scripts/inject_clusters_to_5filters.py** (`--use_white`) to inject those sources onto the HLSP science images; output goes to each filter's `synthetic_fits/` for the following photometry step.

---

## 1. Scripts and entry points

| Script | Role | Called by |
|--------|------|-----------|
| **scripts/run_small_test.py** | Main entry: chains Phase A, Phase B, optional backfill and completeness plots | User runs directly |
| **scripts/generate_white_clusters.py** | Phase A: generate white synthetic images, write `white_position_*`, write **physprop** (from SLUG if no `--input_coords`) | `run_small_test.run_phase_a()` via subprocess |
| **cluster_pipeline.pipeline.pipeline_runner.run_galaxy_pipeline** | Phase B: detection, matching, optional photometry, catalogue; writes labels, match_summary, catalogue | `run_small_test.run_phase_b()` |
| **run_small_test.backfill_physprop_from_white_coords()** | When physprop was not written by generate_white_clusters, fill it from `white_position_*` + optional input_coords | `run_small_test` when `--plot_only` or `--run_photometry` and nframe>1 |
| **run_small_test.plot_completeness_diagnostics()** | Read labels + physprop, plot completeness vs mass / age / mag (5 bands) | `run_small_test` after plot_only or run_photometry |
| **scripts/inject_clusters_to_5filters.py** | Inject matched clusters (same coords) with per-filter mags onto HLSP science images; write `galaxy/filter/synthetic_fits/*.fits` | Pipeline when `inject_5filter_script` is set, after matching and before photometry |

Completeness figures come only from **run_small_test**'s `plot_completeness_diagnostics()`; we do not run `scripts/perform_photometry_ci_cut_on_5filters.py` or `cluster_pipeline.pipeline.diagnostics.plot_completeness_diagnostics` separately.

---

## 2. Typical ways to run

### 2.1 Plot only (no injection / pipeline)

```bash
python scripts/run_small_test.py --plot_only --nframe 10
```

- Does **not** run Phase A or Phase B.
- If `nframe > 1`: runs `backfill_physprop_from_white_coords(nframe)` then `plot_completeness_diagnostics(nframe)`.
- **Assumes**: existing `ngc628-c/white/white_position_{0..nframe-1}_test_reff3.00.txt`; existing `ngc628-c/white/detection_labels/detection_frame{i}_*` or `detection_labels_white_match_frame{i}_*`. Backfill will create/overwrite `physprop/*_frame{i}_*`; mag from last column of `white_position_*`, mass/age from `input_coords` (if 5 columns) or default `ngc628-c/white/input_coords_500.txt` (if present and 5 columns) or SLUG loop.

### 2.2 Full run (injection + detection + matching, no photometry)

```bash
python scripts/run_small_test.py --nframe 1 --ncl 20
# or
python scripts/run_small_test.py --input_coords path/to/x_y_mag.txt --nframe 10
```

- Phase A: `scripts/generate_white_clusters.py` produces white images, `white_position_*`, **physprop** (from SLUG if no `--input_coords`; from generate_white_clusters from the coords file if 5 columns with mass/age).
- Phase B: `run_galaxy_pipeline(max_stage=3)`: detection + matching only; writes `matched_coords`, `match_summary_*`, **white-match labels** (`detection_labels_white_match_*`), and does **not** write `detection_frame_*` (final post-photometry+CI labels).
- No backfill, no plotting (unless you add that separately).

### 2.3 Injection + pipeline through photometry + plotting

```bash
python scripts/run_small_test.py --run_photometry --nframe 10 --ncl 500
# or
python scripts/run_small_test.py --run_photometry --input_coords path/to/5col.txt --nframe 10
```

- Phase A: as above; white images, `white_position_*`, physprop (if generate_white wrote it).
- Phase B: `run_galaxy_pipeline(max_stage=5, run_photometry=True, run_catalogue=True)`. For each frame: after matching, if `inject_5filter_script` is set and physprop mag_VEGA and matched_coords + cluster_ids exist, write per-filter (x,y,mag) coord files (`white/{Filt}_position_{frame}_{outname}_reff{reff}.txt`), then call **inject_clusters_to_5filters.py --use_white** to inject matched clusters onto HLSP 5-filter science images into each filter's `synthetic_fits/`. Then run 5-filter photometry + CI/merr on the "science+injected" FITS for that frame and write catalogue and **detection_frame_*.npy**.
- If nframe>1 and `--input_coords` was passed: then `backfill_physprop_from_white_coords(...)` (may overwrite physprop).
- Finally `plot_completeness_diagnostics(nframe)`.

---

## 3. Data flow and ordering (for plotting)

Plotting assumes **same index = same cluster**:

- **labels**: `detection_frame{i}_*` or `detection_labels_white_match_frame{i}_*`, length = ncl for that frame, order = **cluster_id 0..ncl-1** (injection order).
- **physprop**: `mass_select_*_frame{i}_*`, `age_select_*_frame{i}_*`, `mag_VEGA_select_*_frame{i}_*`, each block length ncl, order = **injection order for that frame** (same as white_position rows).

For multiple frames: concatenate in frame 0, 1, ..., nframe-1 order, so `labels[j]`, `mass[j]`, `age[j]`, `mag_5[j]` refer to the **same cluster** (one row in one frame).

---

## 4. Assumptions and when they break

| Assumption | When it holds | When it breaks |
|------------|----------------|----------------|
| **labels and physprop rows aligned** | Labels and physprop for each frame are written in the same injection order and only from the run_small_test + pipeline chain | If physprop is written by backfill with a **different** input_coords (e.g. 500 rows) while labels are 10x500=5000, backfill still uses `ic_mass[:500]` for frames 1..9, so the same 500 mass/age repeated; only mag is read per frame from white_position, so **mass/age panels are misaligned**; mag panel stays aligned per frame |
| **mag axis = real per-band magnitude** | Using physprop from generate_white with separate synthetic images and mags per filter | With backfill: mag is from the last column of white_position (white light) and **replicated to 5 columns**, so all 5 mag panels are the same white mag; or if run_photometry uses the same white image copied to 5 filters, photometry is done 5 times on the same image and axis labels "F275W/F555W/..." are not real per-band injected mag |
| **completeness vs mag is logistic-like** | Detection probability monotonic in mag and axis is "real band mag" with physical mag bins | Axis is white mag or "fake" 5-band copy; or labels are "after photometry+CI" (bright sources cut by CI/merr); or binning by percentiles scrambles axis/order |
| **physprop from generate_white and 1:1** | No backfill, or backfill uses the same input_coords as the current run (row count = ncl x nframe) | After `--plot_only` or run_photometry with nframe>1, backfill with default `input_coords_500.txt` gives only 500 rows of mass/age, repeated with `ic_mass[:ncl]` for each frame |

---

## 5. Summary: why plots can look wrong

1. **Script chain**: Only `run_small_test.py` is used: Phase A (scripts/generate_white_clusters.py) + Phase B (run_galaxy_pipeline) + optional backfill + `plot_completeness_diagnostics()`. We do not use scripts/perform_photometry_ci_cut_on_5filters or the pipeline's other diagnostics for plotting.
2. **Five filter panels**: Either backfill **replicates white mag to 5 columns** (five mag panels share the same x), or photometry runs on **the same white image copied to 5 filters**; in both cases the horizontal axis is not "real per-band injected magnitude", so completeness vs mag need not (and often does not) look like a clean logistic.
3. **mass/age**: For nframe>1 with backfill using 500-row input_coords, only frame 0 has mass/age aligned with labels; frames 1..9 reuse the same 500 rows, so **misaligned**.
4. **Binning**: Mag bins are fixed magnitude bins with x sorted; if data are misaligned or the axis is not "real band mag", curve shapes can still look odd.

For "correct" completeness vs mag (logistic shape, clear physical meaning) you need either:
- Plot only **vs white mag** (one panel), with both mag and labels from the same run and same order; or
- Use **per-filter** synthetic images and injected mags, run photometry, then plot using the corresponding per-band mag from physprop (five separate panels).
