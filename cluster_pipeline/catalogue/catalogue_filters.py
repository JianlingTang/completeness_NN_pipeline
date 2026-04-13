"""
Catalogue filters per LEGUS: sources detected in at least four bands with photometric
error below 0.3 mag, and absolute V-band magnitude brighter than -6 mag. The V-band
magnitude cut uses V-band photometry from the CI-based (science) aperture.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schemas import CATALOGUE_FILTERS_SCHEMA

# Canonical filter names for LEGUS criteria (same order as physprop mag columns: 275, 336, 435, 555, 814)
VBAND_FILTER = "F555W"
B_FILTER = "F435W"
I_FILTER = "F814W"

MERR_CUT = 0.3
# LEGUS: discard if M_V > -6 → keep M_V <= -6 (keep bright, discard faint).
# Apparent V turnover: m_V = dmod + M_V → at M_V = -6, m_V = dmod - 6 (e.g. 23.98 for dmod=29.98).
MV_CUT = -6  # keep only M_V <= -6 (turnover at apparent V ≈ dmod - 6 mag)


def apply_catalogue_filters(
    photometry_parquet_path: Path,
    merr_cut: float = MERR_CUT,
    vband_filter: str = VBAND_FILTER,
    b_filter: str = B_FILTER,
    i_filter: str = I_FILTER,
    dmod: float | None = None,
) -> pd.DataFrame:
    """
    At least four bands with merr <= merr_cut (0.3 mag); M_V <= -6 (V from science aperture).
    CI cut from photometry (V-band CI >= threshold). No requirement that V be one of the four.
    """
    df = pd.read_parquet(photometry_parquet_path)
    if "passes_ci" not in df.columns:
        df["passes_ci"] = 1
    # Match filter names case-insensitively (photometry may have 'f555w' vs 'F555W')
    df["_fn"] = df["filter_name"].astype(str).str.upper()
    v_upper = vband_filter.upper()
    b_upper = b_filter.upper()
    i_upper = i_filter.upper()
    grp = df.groupby(["cluster_id", "galaxy_id", "frame_id", "reff"])

    agg = grp.agg(passes_ci=("passes_ci", "max")).reset_index()

    # At least four bands (any of the 5) with photometric error below merr_cut (0.3 mag).
    agg["passes_stage1_merr"] = 1
    n_good = grp.apply(lambda g: (g["merr"] <= merr_cut).sum()).values
    agg["passes_stage2_merr"] = (n_good >= 4).astype(np.int8)

    # Stage B M_V <= -6
    if dmod is not None:
        def v_mag_from_g(g: pd.DataFrame):
            v = g[g["_fn"] == v_upper]
            return v["mag"].iloc[0] if len(v) else np.nan
        v_mag = grp.apply(v_mag_from_g).values
        M_V = v_mag - dmod
        agg["passes_MV"] = (M_V <= MV_CUT).astype(np.int8)
    else:
        agg["passes_MV"] = 1

    agg["in_catalogue"] = (
        (agg["passes_ci"] == 1)
        & (agg["passes_stage1_merr"] == 1)
        & (agg["passes_stage2_merr"] == 1)
        & (agg["passes_MV"] == 1)
    ).astype(np.int8)
    return agg[list(CATALOGUE_FILTERS_SCHEMA.keys())]


def write_catalogue_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write catalogue table to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
