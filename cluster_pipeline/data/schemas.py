"""
Explicit schemas for intermediate parquet tables.
All pipeline stages read/write parquet with these column names and dtypes.
Enables cluster_id-based joins and reproducible data flow.
"""

# Schema definitions: column name -> dtype (pandas/numpy style)
# Used for validation and when writing parquet.

INJECTED_CLUSTERS_SCHEMA: dict[str, str] = {
    "cluster_id": "int64",
    "galaxy_id": "str",
    "frame_id": "int32",
    "reff": "float64",
    "mass": "float64",
    "age": "float64",
    "av": "float64",
    "radius": "float64",
    "x": "float64",
    "y": "float64",
}
# Optional magnitude columns: mag_f1, mag_f2, ... (filter-dependent)
INJECTED_CLUSTERS_MAG_COLUMNS = "mag_f"  # prefix; actual: mag_f0, mag_f1, ...

MATCH_RESULTS_SCHEMA: dict[str, str] = {
    "cluster_id": "int64",
    "galaxy_id": "str",
    "frame_id": "int32",
    "reff": "float64",
    "matched": "int8",  # 0 or 1
    "injected_x": "float64",
    "injected_y": "float64",
    "detected_x": "float64",
    "detected_y": "float64",
}

PHOTOMETRY_SCHEMA: dict[str, str] = {
    "cluster_id": "int64",
    "galaxy_id": "str",
    "frame_id": "int32",
    "reff": "float64",
    "filter_name": "str",
    "mag": "float64",
    "merr": "float64",
    "mag_1px": "float64",
    "mag_3px": "float64",
    "ci": "float64",  # mag_1px - mag_3px
}

CATALOGUE_FILTERS_SCHEMA: dict[str, str] = {
    "cluster_id": "int64",
    "galaxy_id": "str",
    "frame_id": "int32",
    "reff": "float64",
    "passes_ci": "int8",
    "passes_stage1_merr": "int8",   # Stage A: V<=0.3 and (B<=0.3 or I<=0.3)
    "passes_stage2_merr": "int8",   # Stage B: at least 4 bands merr<=0.3
    "passes_MV": "int8",            # Stage B: M_V <= -6            # M_V <= -6 (if dmod given)
    "in_catalogue": "int8",
}

DATASET_ROW_SCHEMA: dict[str, str] = {
    "cluster_id": "int64",
    "galaxy_id": "str",
    "frame_id": "int32",
    "reff": "float64",
    "mass": "float64",
    "age": "float64",
    "av": "float64",
    "mag_f0": "float64",
    "mag_f1": "float64",
    "mag_f2": "float64",
    "mag_f3": "float64",
    "mag_f4": "float64",
    "detection_label": "int8",  # C in {0, 1}
}


def get_required_columns(schema_name: str) -> list[str]:
    """Return ordered list of required columns for a schema."""
    schemas = {
        "injected_clusters": INJECTED_CLUSTERS_SCHEMA,
        "match_results": MATCH_RESULTS_SCHEMA,
        "matched_clusters": MATCH_RESULTS_SCHEMA,  # alias for CI/dataset naming
        "photometry": PHOTOMETRY_SCHEMA,
        "catalogue_filters": CATALOGUE_FILTERS_SCHEMA,
        "dataset_row": DATASET_ROW_SCHEMA,
        "labels": DATASET_ROW_SCHEMA,  # alias: dataset_row carries detection_label
    }
    s = schemas.get(schema_name)
    return list(s.keys()) if s else []
