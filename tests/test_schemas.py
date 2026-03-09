"""
Tests for cluster_pipeline.data.schemas.
Focus: schema consistency, required columns, dataset row alignment.
"""
import pytest

from cluster_pipeline.data.schemas import (
    CATALOGUE_FILTERS_SCHEMA,
    DATASET_ROW_SCHEMA,
    INJECTED_CLUSTERS_SCHEMA,
    MATCH_RESULTS_SCHEMA,
    PHOTOMETRY_SCHEMA,
    get_required_columns,
)


class TestSchemaDefinitions:
    """All schemas must include cluster_id for joins."""

    def test_injected_clusters_has_cluster_id(self):
        assert "cluster_id" in INJECTED_CLUSTERS_SCHEMA
        assert INJECTED_CLUSTERS_SCHEMA["cluster_id"] == "int64"

    def test_match_results_has_cluster_id(self):
        assert "cluster_id" in MATCH_RESULTS_SCHEMA

    def test_photometry_has_cluster_id(self):
        assert "cluster_id" in PHOTOMETRY_SCHEMA

    def test_catalogue_filters_has_cluster_id(self):
        assert "cluster_id" in CATALOGUE_FILTERS_SCHEMA

    def test_dataset_row_has_cluster_id_and_detection_label(self):
        assert "cluster_id" in DATASET_ROW_SCHEMA
        assert "detection_label" in DATASET_ROW_SCHEMA
        assert DATASET_ROW_SCHEMA["detection_label"] == "int8"


class TestGetRequiredColumns:
    """get_required_columns returns ordered list matching schema keys."""

    @pytest.mark.parametrize("schema_name", [
        "injected_clusters",
        "match_results",
        "matched_clusters",
        "photometry",
        "catalogue_filters",
        "dataset_row",
        "labels",
    ])
    def test_returns_list_for_known_schema(self, schema_name):
        cols = get_required_columns(schema_name)
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "cluster_id" in cols

    def test_dataset_row_columns_order_and_mag_f0_to_f4(self):
        cols = get_required_columns("dataset_row")
        for i in range(5):
            assert f"mag_f{i}" in cols
        assert cols[0] == "cluster_id"

    def test_unknown_schema_returns_empty_list(self):
        assert get_required_columns("unknown_schema") == []

    def test_required_columns_match_schema_keys(self):
        schemas = {
            "injected_clusters": INJECTED_CLUSTERS_SCHEMA,
            "match_results": MATCH_RESULTS_SCHEMA,
            "photometry": PHOTOMETRY_SCHEMA,
            "catalogue_filters": CATALOGUE_FILTERS_SCHEMA,
            "dataset_row": DATASET_ROW_SCHEMA,
        }
        for name, schema in schemas.items():
            cols = get_required_columns(name)
            assert set(cols) == set(schema.keys())
            assert len(cols) == len(schema)

    def test_ci_dataset_schemas_have_cluster_id_and_required_columns(self):
        """CI: injected_clusters, matched_clusters, photometry, labels all have cluster_id and non-empty columns."""
        ci_schemas = ["injected_clusters", "matched_clusters", "photometry", "labels"]
        for name in ci_schemas:
            cols = get_required_columns(name)
            assert "cluster_id" in cols, f"CI schema {name} must include cluster_id"
            assert len(cols) > 0, f"CI schema {name} must have required columns"
        assert get_required_columns("labels") == get_required_columns("dataset_row")
        assert get_required_columns("matched_clusters") == get_required_columns("match_results")
