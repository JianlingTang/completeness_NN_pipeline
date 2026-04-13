#!/usr/bin/env python3
"""
Experiment wrapper: natural **missing F275W** while catalogue-retained (Y4_F275W).

**Assumptions (read before trusting results)**

1. **Default pipeline only**: all science runs use ``python scripts/run_pipeline.py`` as a
   subprocess with **no edits** to that file.

2. **Y4_F275W label** (implemented in ``compute_y4_f275w``): uses only pipeline products
   ``match_results_*.parquet``, ``catalogue_*.parquet``, ``photometry_*.parquet``.
   *Recovered* means ``matched==1`` and ``in_catalogue==1``.
   *Missing F275W* means: F275W row is **not** ``band_usable`` (see function docstring), and
   F336W, F435W, F555W, F814W are **all** ``band_usable``. Any other missing-filter pattern → 0.

3. **Baseline experiment**: calls ``run_pipeline.py`` **without** ``--input_coords`` — identical
   injection to a normal user run (SLUG placement + sampling inside ``generate_white_clusters``).

4. **Targeted experiment**: cannot change ``generate_white_clusters`` / pipeline internals.
   We only change the injected population by writing ``--input_coords`` (5 columns:
   ``x y mag mass age`` in **column, row** order, same as ``sample_slug_white_mag.py``).
   *Placement* uses the same σ_pc-convolved white image PDF as ``generate_white_clusters.py``
   (code duplicated here — if that upstream routine changes, re-sync this block).
   *mag* (3rd column) uses the same white-light mag formula as ``sample_slug_white_mag.py``.

5. **Reweighting**: targeted draws are biased; for population inference you must reweight
   back to the baseline SLUG prior (not implemented here — only counts and diagnostics).

Usage (examples)
----------------
  # Baseline (default SLUG injection; full pipeline)
  python scripts/experiment_missing_f275w.py baseline --galaxy ngc628-e --ncl 200 --nframe 1 \\
      --reff-list 5 --outname exp_y4_base --skip-galaxy-setup

  # Targeted biased draw + coords file + same pipeline
  python scripts/experiment_missing_f275w.py targeted --galaxy ngc628-e --scheme uv_faint \\
      --ncl 200 --nframe 1 --reff-list 5 --outname exp_y4_uv --skip-galaxy-setup

  # Summarise a finished run (reads parquet under <galaxy>/white/catalogue/)
  python scripts/experiment_missing_f275w.py analyze --galaxy ngc628-e --outname exp_y4_base \\
      --nframe 1 --reff-list 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_PIPELINE = ROOT / "scripts" / "run_pipeline.py"

CANONICAL_5 = ("F275W", "F336W", "F435W", "F555W", "F814W")
MAG_BAD = 90.0  # INDEF → 99.999 in aperture txt; treat >=90 as unusable


def _norm_filt(s: str) -> str:
    return str(s).strip().upper()


def band_usable_row(mag: float, merr: float, passes_merr: int) -> bool:
    """Usable photometry: finite mag, not INDEF sentinel, and passes_merr flag set."""
    if not np.isfinite(mag) or not np.isfinite(merr):
        return False
    if float(mag) >= MAG_BAD:
        return False
    return int(passes_merr) == 1


def compute_y4_f275w(
    matched: int,
    in_catalogue: int,
    phot_cluster: pd.DataFrame,
) -> int:
    """
    Y4_F275W = 1 iff ALL of:
      - matched == 1 (white detection association)
      - in_catalogue == 1 (final catalogue retention; same cuts as default pipeline)
      - F275W photometry row exists but is **not** band_usable
      - F336W, F435W, F555W, F814W rows exist and **are** band_usable

    Everything else → 0 (including other 4-filter patterns, 5-filter, or non-retained).
    """
    if int(matched) != 1 or int(in_catalogue) != 1:
        return 0
    if phot_cluster is None or phot_cluster.empty:
        return 0
    by_f: dict[str, pd.Series] = {}
    for _, r in phot_cluster.iterrows():
        fn = _norm_filt(r["filter_name"])
        by_f[fn] = r
    for need in CANONICAL_5:
        if need not in by_f:
            return 0
    f275 = by_f["F275W"]
    others = ("F336W", "F435W", "F555W", "F814W")
    if not all(
        band_usable_row(float(by_f[f]["mag"]), float(by_f[f]["merr"]), int(by_f[f]["passes_merr"]))
        for f in others
    ):
        return 0
    if band_usable_row(float(f275["mag"]), float(f275["merr"]), int(f275["passes_merr"])):
        return 0
    return 1


def generate_white_light(scale_factors, f275, f336, f438, f555, f814):
    """Same as ``sample_slug_white_mag.generate_white_light`` (white combination)."""
    scale_factors = np.asarray(scale_factors) / scale_factors[2]
    denom = np.sqrt(np.sum((1.0 / scale_factors) ** 2))
    white = (
        np.sqrt(
            (f275 / scale_factors[0]) ** 2
            + (f336 / scale_factors[1]) ** 2
            + (f438 / scale_factors[2]) ** 2
            + (f555 / scale_factors[3]) ** 2
            + (f814 / scale_factors[4]) ** 2
        )
        / denom
    )
    return white


def _slug_read_filters(main_dir: Path, galaxy: str) -> list[str]:
    gal_filters = np.load(main_dir / "galaxy_filter_dict.npy", allow_pickle=True).item()
    key = galaxy if galaxy in gal_filters else galaxy.split("_")[0]
    if key not in gal_filters:
        raise KeyError(f"galaxy {galaxy!r} not in galaxy_filter_dict (tried {key!r})")
    filters, cams = gal_filters[key]
    allfilters_cam: list[str] = []
    for filt, cam in zip(filters, cams):
        f, c = filt.upper(), cam.upper()
        if c == "WFC3":
            c = "WFC3_UVIS"
        allfilters_cam.append(f"{c}_{f}")
    return sorted(allfilters_cam, key=lambda x: x[-4:])


def _mag_bao_from_phot_neb_ex(phot_neb_ex: np.ndarray, main_dir: Path, galaxy: str, dmod: float) -> np.ndarray:
    """Vector mag_BAO for all clusters (same recipe as sample_slug_white_mag / generate_white_clusters)."""
    import astropy.units as u
    import glob
    from astropy.coordinates import Distance
    from astropy.io import fits

    from cluster_pipeline.data.slug_reader import read_cluster

    gal_filters = np.load(main_dir / "galaxy_filter_dict.npy", allow_pickle=True).item()
    key = galaxy if galaxy in gal_filters else galaxy.split("_")[0]
    filters = sorted(gal_filters[key][0])
    headers = {}
    for fn in filters:
        pat = str(main_dir / galaxy / f"*{fn}*drc.fits")
        matches = glob.glob(pat)
        if not matches:
            raise FileNotFoundError(f"No FITS for filter {fn}: {pat}")
        _, headers[fn] = fits.getdata(matches[0], header=True)
    scaling_factors = np.array([headers[fn]["PHOTFLAM"] for fn in filters])
    d = Distance(distmod=dmod * u.mag)
    distance_in_cm = d.to(u.cm).value
    phot_f = phot_neb_ex / (4 * np.pi * distance_in_cm**2)
    scaled = phot_f / scaling_factors
    f275 = scaled[:, 0]
    f336 = scaled[:, 1]
    f438 = scaled[:, 2]
    f555 = scaled[:, 3]
    f814 = scaled[:, 4]
    img21 = generate_white_light([55.8, 45.3, 44.2, 65.7, 29.3], f275, f336, f438, f555, f814)
    img24 = generate_white_light([0.5, 0.8, 3.1, 11.2, 13.7], f275, f336, f438, f555, f814)
    flux = 0.5 * (img21 + img24)
    return np.log10(np.maximum(flux, 1e-300)) / -0.4


def _load_slug_full(main_dir: Path, galaxy: str, dmod: float):
    from cluster_pipeline.data.slug_reader import read_cluster

    libname = str(main_dir / "SLUG_library" / "flat_in_logm")
    rf = _slug_read_filters(main_dir, galaxy)
    lib = read_cluster(libname, read_filters=rf, photsystem="L_lambda")
    phot_veg = np.asarray(lib.phot_neb_ex_veg, dtype=float) + float(dmod)
    mag_bao = _mag_bao_from_phot_neb_ex(np.asarray(lib.phot_neb_ex, dtype=float), main_dir, galaxy, dmod)
    mass = np.asarray(lib.actual_mass, dtype=float)
    age = np.maximum(np.asarray(lib.time, dtype=float) - np.asarray(lib.form_time, dtype=float), 1e6)
    av_raw = getattr(lib, "a_v", None)
    if av_raw is None:
        av = np.zeros_like(mass, dtype=float)
    else:
        av = np.asarray(av_raw, dtype=float)
        if av.shape != mass.shape:
            av = np.zeros_like(mass, dtype=float)
    n = phot_veg.shape[0]
    return lib, phot_veg, mag_bao, mass, age, av, rf, n


def _placement_convolved(
    white_fits: Path,
    sigma_pc: float,
    galdist_pc: float,
    reff_pc: float,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample (x_col, y_row) indices; PDF ∝ Gaussian-smoothed white flux on pixels with
    positive flux in both raw and smoothed images (mirrors generate_white_clusters logic).
    """
    import astropy.units as u
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter

    pixscale_wfc3 = 0.039622  # arcsec/px; same default branch as white sciframe in generate_white_clusters
    hdul = fits.open(white_fits)
    try:
        image_data = np.asarray(hdul[0].data, dtype=float)
    finally:
        hdul.close()
    image_shape = image_data.shape
    image_data_flat = image_data.ravel()
    galdist = float(galdist_pc)
    theta = np.arctan(sigma_pc * u.pc / (galdist * u.pc))
    theta = theta.to(u.arcsec)
    pix_scale = pixscale_wfc3 * u.arcsec / u.pixel
    pix_cl = (sigma_pc / (2 * np.pi * galdist) * 3600) / float(pix_scale.to(u.arcsec / u.pixel).value)
    filtered = gaussian_filter(image_data, sigma=pix_cl)
    filtered_flat = filtered.ravel()
    positive_indices_image = np.where(image_data_flat > 0)
    positive_indices_filtered = np.where(filtered_flat > 0)
    image_data_positive = np.zeros_like(image_data_flat, dtype=bool)
    filtered_positive = np.zeros_like(filtered_flat, dtype=bool)
    image_data_positive[positive_indices_image] = True
    filtered_positive[positive_indices_filtered] = True
    p_id = np.logical_and(image_data_positive, filtered_positive)
    positive_indices_result = np.where(p_id)[0]
    if positive_indices_result.size == 0:
        raise ValueError("No valid placement pixels on white image.")
    placement_weights = np.maximum(filtered_flat[positive_indices_result].astype(np.float64), 0.0)
    sw = placement_weights.sum()
    if sw <= 0:
        raise ValueError("Placement weights sum to zero.")
    placement_weights /= sw

    rng = np.random.default_rng(seed)
    # minsep branch omitted (generate default minsep=False)
    xs = np.zeros(n, dtype=int)
    ys = np.zeros(n, dtype=int)
    for k in range(n):
        flat_ix = int(rng.choice(positive_indices_result, p=placement_weights))
        rr, cc = np.unravel_index(flat_ix, image_shape)
        # generate_white_clusters: x,y = unravel then append (y,x) → file col,row = cc, rr? see sample_slug: x=col
        # unravel returns (axis0, axis1) = (row, col) for C-order
        row_i, col_i = int(rr), int(cc)
        xs[k] = col_i
        ys[k] = row_i
    return xs, ys


def _target_weights(scheme: str, phot_veg: np.ndarray, mag_bao: np.ndarray, mass: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Non-negative unnormalized weights over SLUG rows for ``scheme``."""
    m275 = phot_veg[:, 0]
    logm = np.log10(np.maximum(mass, 1e-20))
    loga = np.log10(np.maximum(age, 1e6))
    if scheme == "uv_faint":
        # Favor intrinsically faint UV (higher Vega mag) — often near F275W dropout
        return np.exp(0.35 * (m275 - np.median(m275)))
    if scheme == "old_age":
        return np.exp(0.5 * (loga - np.median(loga)))
    if scheme == "low_mass":
        return np.exp(-1.2 * (logm - np.median(logm)))
    if scheme == "mixed":
        w1 = _target_weights("uv_faint", phot_veg, mag_bao, mass, age)
        w2 = _target_weights("old_age", phot_veg, mag_bao, mass, age)
        return w1 * w2
    raise ValueError(f"Unknown scheme: {scheme}")


def write_targeted_input_coords(
    main_dir: Path,
    galaxy: str,
    dmod: float,
    sigma_pc: float,
    galdist_pc: float,
    reff_pc: float,
    n: int,
    scheme: str,
    seed: int,
    out_path: Path,
    manifest_path: Path,
) -> None:
    _, phot_veg, mag_bao, mass, age, av, _, ntot = _load_slug_full(main_dir, galaxy, dmod)
    w = _target_weights(scheme, phot_veg, mag_bao, mass, age)
    w = np.maximum(w, 1e-300)
    w /= w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(ntot, size=n, replace=n > ntot, p=w)
    mag_sel = mag_bao[idx]
    mass_sel = mass[idx]
    age_sel = age[idx]
    av_sel = av[idx]
    phot_sel = phot_veg[idx, :]

    white_fits = main_dir / galaxy / f"{galaxy}_white.fits"
    if not white_fits.is_file():
        raise FileNotFoundError(white_fits)
    xs, ys = _placement_convolved(white_fits, sigma_pc, galdist_pc, reff_pc, n, seed + 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for x, y, mb, m, a in zip(xs, ys, mag_sel, mass_sel, age_sel):
            f.write(f"{int(x)} {int(y)} {float(mb)} {float(m)} {float(a)}\n")

    manifest = pd.DataFrame(
        {
            "slug_row": idx,
            "log_mass": np.log10(np.maximum(mass_sel, 1e-20)),
            "log_age_yr": np.log10(np.maximum(age_sel, 1e6)),
            "av": av_sel,
            "mag_bao": mag_sel,
            "m275_vega": phot_sel[:, 0],
            "m336_vega": phot_sel[:, 1],
            "m435_vega": phot_sel[:, 2],
            "m555_vega": phot_sel[:, 3],
            "m814_vega": phot_sel[:, 4],
            "scheme": scheme,
            "placement": "convolved_white_pdf",
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)


def run_pipeline_unchanged(
    python_exe: str,
    main_dir: Path,
    galaxy: str,
    outname: str,
    nframe: int,
    ncl: int,
    reff_list: list[float],
    input_coords: Path | None,
    skip_galaxy_setup: bool,
    sigma_pc: float,
    no_photometry: bool = False,
) -> int:
    cmd = [
        python_exe,
        str(RUN_PIPELINE),
        "--galaxy",
        galaxy,
        "--outname",
        outname,
        "--nframe",
        str(nframe),
        "--ncl",
        str(ncl),
        "--reff_list",
        ",".join(str(int(r)) if abs(r - int(r)) < 1e-9 else str(r) for r in reff_list),
        "--sigma_pc",
        str(sigma_pc),
    ]
    if skip_galaxy_setup:
        cmd.append("--skip-galaxy-setup")
    if no_photometry:
        cmd.append("--no_photometry")
    if input_coords is not None:
        cmd.extend(["--input_coords", str(input_coords.resolve())])
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(main_dir)).returncode


def build_per_cluster_table(
    main_dir: Path,
    galaxy: str,
    outname: str,
    nframe: int,
    reff_list: list[float],
) -> pd.DataFrame:
    cat_dir = main_dir / galaxy / "white" / "catalogue"
    phys_dir = main_dir / "physprop"
    rows: list[dict] = []
    mrmodel = "flat"
    for i_frame in range(nframe):
        for reff in reff_list:
            mp = cat_dir / f"match_results_frame{i_frame}_{outname}_reff{float(reff):.2f}.parquet"
            cp = cat_dir / f"catalogue_frame{i_frame}_{outname}_reff{float(reff):.2f}.parquet"
            pp = cat_dir / f"photometry_frame{i_frame}_{outname}_reff{float(reff):.2f}.parquet"
            if not (mp.is_file() and cp.is_file() and pp.is_file()):
                print(f"WARNING: missing outputs for frame={i_frame} reff={reff}", flush=True)
                continue
            match_df = pd.read_parquet(mp)
            cat_df = pd.read_parquet(cp)
            phot_df = pd.read_parquet(pp)
            cat_by_cid = {int(r["cluster_id"]): int(r["in_catalogue"]) for _, r in cat_df.iterrows()}
            base = f"reff{int(float(reff))}_{outname}"
            mass_npy = phys_dir / f"mass_select_model{mrmodel}_frame{i_frame}_{base}.npy"
            age_npy = phys_dir / f"age_select_model{mrmodel}_frame{i_frame}_{base}.npy"
            av_npy = phys_dir / f"av_select_model{mrmodel}_frame{i_frame}_{base}.npy"
            mag_npy = phys_dir / f"mag_VEGA_select_model{mrmodel}_frame{i_frame}_{base}.npy"
            mass_arr = np.load(mass_npy) if mass_npy.is_file() else None
            age_arr = np.load(age_npy) if age_npy.is_file() else None
            av_arr = np.load(av_npy) if av_npy.is_file() else None
            mag_arr = np.load(mag_npy) if mag_npy.is_file() else None

            for _, mrow in match_df.iterrows():
                cid = int(mrow["cluster_id"])
                matched = int(mrow["matched"])
                ix = float(mrow["injected_x"])
                iy = float(mrow["injected_y"])
                incat = int(cat_by_cid.get(cid, 0))
                psub = phot_df.loc[phot_df["cluster_id"] == cid]
                y4 = compute_y4_f275w(matched, incat, psub)
                rec: dict = {
                    "galaxy": galaxy,
                    "frame_id": i_frame,
                    "reff": float(reff),
                    "cluster_id": cid,
                    "matched": matched,
                    "in_catalogue": incat,
                    "Y4_F275W": y4,
                    "injected_x": ix,
                    "injected_y": iy,
                }
                if mass_arr is not None and cid < len(mass_arr):
                    rec["mass"] = float(mass_arr[cid])
                    rec["log_mass"] = float(np.log10(max(mass_arr[cid], 1e-20)))
                if age_arr is not None and cid < len(age_arr):
                    rec["age_yr"] = float(age_arr[cid])
                    rec["log_age_yr"] = float(np.log10(max(age_arr[cid], 1e6)))
                if av_arr is not None and cid < len(av_arr):
                    rec["av"] = float(av_arr[cid])
                if mag_arr is not None and cid < len(mag_arr):
                    mv = mag_arr[cid] if mag_arr.ndim == 1 else mag_arr[cid, :]
                    mv = np.atleast_1d(np.asarray(mv, dtype=float)).ravel()
                    for j, fn in enumerate(CANONICAL_5):
                        if j < len(mv):
                            rec[f"mag_vega_{fn}"] = float(mv[j])
                rows.append(rec)
    return pd.DataFrame(rows)


def analyze_and_save(
    main_dir: Path,
    galaxy: str,
    outname: str,
    nframe: int,
    reff_list: list[float],
    experiment_tag: str,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = build_per_cluster_table(main_dir, galaxy, outname, nframe, reff_list)
    out_root = main_dir / "experiments" / "missing_f275w" / experiment_tag
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_root / "per_cluster_labels.parquet", index=False)

    n = len(df)
    pos = int(df["Y4_F275W"].sum()) if n else 0
    frac = pos / n if n else 0.0
    summary = {
        "n_clusters": n,
        "Y4_pos": pos,
        "Y4_frac": frac,
        "imbalance_neg_per_pos": (n - pos) / pos if pos else None,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Binned p(Y4|log_mass) if column present
    if n and "log_mass" in df.columns:
        edges = np.linspace(df["log_mass"].min(), df["log_mass"].max(), 12)
        df = df.copy()
        df["_lm_bin"] = pd.cut(df["log_mass"], edges, include_lowest=True)
        g = df.groupby("_lm_bin", observed=False)["Y4_F275W"].agg(["mean", "count"]).reset_index()
        g.to_csv(out_root / "binned_Y4_vs_log_mass.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        centers = []
        for iv in g["_lm_bin"]:
            if iv is not None and not (isinstance(iv, float) and np.isnan(iv)):
                try:
                    centers.append(float(iv.mid))
                except Exception:
                    centers.append(np.nan)
            else:
                centers.append(np.nan)
        ax.plot(centers, g["mean"], "o-")
        ax.set_xlabel(r"$\log_{10}$(mass / M$_\odot$)")
        ax.set_ylabel(r"$\mathbb{P}(Y4=1)$ (binned mean)")
        ax.set_title(f"{galaxy} {outname}: Y4_F275W vs mass")
        fig.tight_layout()
        fig.savefig(out_root / "Y4_vs_log_mass.png", dpi=140)
        plt.close(fig)

    if n and "log_mass" in df.columns and "log_age_yr" in df.columns:
        # 2D mean heatmap via pivot
        lm = df["log_mass"].values
        la = df["log_age_yr"].values
        y4 = df["Y4_F275W"].values
        nx, ny = 16, 16
        xe = np.linspace(np.nanmin(lm), np.nanmax(lm), nx + 1)
        ye = np.linspace(np.nanmin(la), np.nanmax(la), ny + 1)
        grid = np.full((ny, nx), np.nan)
        cnt = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                m = (lm >= xe[i]) & (lm < xe[i + 1]) & (la >= ye[j]) & (la < ye[j + 1])
                if np.any(m):
                    grid[j, i] = float(np.mean(y4[m]))
                    cnt[j, i] = int(np.sum(m))
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="mean Y4")
        ax.set_xlabel(r"$\log_{10}$(mass)")
        ax.set_ylabel(r"$\log_{10}$(age / yr)")
        ax.set_title(f"{galaxy} {outname}: mean Y4 in (log M, log age) bins")
        fig.tight_layout()
        fig.savefig(out_root / "Y4_heatmap_mass_age.png", dpi=140)
        plt.close(fig)

    # Rank point-biserial correlation with Y4
    corr_rows = []
    if n:
        y = df["Y4_F275W"].astype(float).values
        for col in df.columns:
            if col in ("Y4_F275W", "galaxy", "cluster_id", "matched", "in_catalogue"):
                continue
            if df[col].dtype == object:
                continue
            x = pd.to_numeric(df[col], errors="coerce").values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            if np.isfinite(r):
                corr_rows.append((col, float(r)))
    corr_rows.sort(key=lambda t: abs(t[1]), reverse=True)
    pd.DataFrame(corr_rows, columns=["feature", "pearson_r_vs_Y4"]).to_csv(
        out_root / "feature_correlations.csv", index=False
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis under {out_root}")
    return summary


def cmd_baseline(ns: argparse.Namespace) -> int:
    return run_pipeline_unchanged(
        sys.executable,
        Path(ns.main_dir).resolve(),
        ns.galaxy,
        ns.outname,
        ns.nframe,
        ns.ncl,
        [float(x) for x in ns.reff_list],
        None,
        ns.skip_galaxy_setup,
        ns.sigma_pc,
    )


def cmd_targeted(ns: argparse.Namespace) -> int:
    main_dir = Path(ns.main_dir).resolve()
    coords = main_dir / "experiments" / "missing_f275w" / "_coords_cache" / f"{ns.outname}_input_coords.txt"
    manifest = main_dir / "experiments" / "missing_f275w" / "_coords_cache" / f"{ns.outname}_manifest.parquet"
    write_targeted_input_coords(
        main_dir=main_dir,
        galaxy=ns.galaxy,
        dmod=ns.dmod,
        sigma_pc=ns.sigma_pc,
        galdist_pc=ns.galdist_pc,
        reff_pc=float(ns.reff_list[0]),
        n=ns.ncl,
        scheme=ns.scheme,
        seed=ns.seed,
        out_path=coords,
        manifest_path=manifest,
    )
    return run_pipeline_unchanged(
        sys.executable,
        main_dir,
        ns.galaxy,
        ns.outname,
        ns.nframe,
        ns.ncl,
        [float(x) for x in ns.reff_list],
        coords,
        ns.skip_galaxy_setup,
        ns.sigma_pc,
    )


def cmd_analyze(ns: argparse.Namespace) -> int:
    analyze_and_save(
        Path(ns.main_dir).resolve(),
        ns.galaxy,
        ns.outname,
        ns.nframe,
        [float(x) for x in ns.reff_list],
        experiment_tag=ns.experiment_tag or ns.outname,
    )
    return 0


def cmd_recommend(ns: argparse.Namespace) -> int:
    """Print strategy recommendation by comparing two summary.json paths if given."""
    print(
        "\n--- Recommendation (manual review of summary.json + plots) ---\n"
        "1. Prefer regions where binned mean Y4 rises: typically faint m275_vega, older ages, "
        "and lower masses in our biased schemes — confirm on YOUR galaxy with heatmaps.\n"
        "2. Targeted q(θ): use ``targeted --scheme mixed`` or ``uv_faint`` for higher positives; "
        "keep a 20–40% baseline uniform slab for boundary coverage (run separate baseline job).\n"
        "3. For NN training: concatenate baseline + targeted manifests with inverse-probability weights; "
        "single targeted pass alone is usually insufficient for calibration.\n"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Missing-F275W catalogue experiment wrapper (does not modify run_pipeline.py).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("baseline", help="Run default run_pipeline.py (no input_coords)")
    p0.add_argument("--main-dir", type=Path, default=ROOT)
    p0.add_argument("--galaxy", required=True)
    p0.add_argument("--outname", required=True)
    p0.add_argument("--nframe", type=int, default=1)
    p0.add_argument("--ncl", type=int, required=True)
    p0.add_argument("--reff-list", nargs="+", type=float, required=True)
    p0.add_argument("--skip-galaxy-setup", action="store_true")
    p0.add_argument("--sigma-pc", type=float, default=100.0)
    p0.set_defaults(func=cmd_baseline)

    p1 = sub.add_parser("targeted", help="Biased SLUG draw + input_coords + same run_pipeline.py")
    p1.add_argument("--main-dir", type=Path, default=ROOT)
    p1.add_argument("--galaxy", required=True)
    p1.add_argument("--outname", required=True)
    p1.add_argument("--nframe", type=int, default=1)
    p1.add_argument("--ncl", type=int, required=True)
    p1.add_argument("--reff-list", nargs="+", type=float, required=True)
    p1.add_argument("--scheme", choices=("uv_faint", "old_age", "low_mass", "mixed"), default="uv_faint")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--dmod", type=float, default=29.98)
    p1.add_argument("--galdist-pc", type=float, default=7.2e6, help="Galaxy distance in pc (NGC628 ~7.2 Mpc)")
    p1.add_argument("--skip-galaxy-setup", action="store_true")
    p1.add_argument("--sigma-pc", type=float, default=100.0)
    p1.set_defaults(func=cmd_targeted)

    p2 = sub.add_parser("analyze", help="Parse pipeline outputs and write Y4 diagnostics")
    p2.add_argument("--main-dir", type=Path, default=ROOT)
    p2.add_argument("--galaxy", required=True)
    p2.add_argument("--outname", required=True)
    p2.add_argument("--nframe", type=int, default=1)
    p2.add_argument("--reff-list", nargs="+", type=float, required=True)
    p2.add_argument("--experiment-tag", type=str, default=None)
    p2.set_defaults(func=cmd_analyze)

    p3 = sub.add_parser("recommend", help="Print sampling strategy notes")
    p3.set_defaults(func=cmd_recommend)

    ns = ap.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
