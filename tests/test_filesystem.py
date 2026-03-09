"""
Tests for cluster_pipeline.utils.filesystem.
Focus: path handling, directory creation, safe cleanup.
"""

from cluster_pipeline.utils.filesystem import (
    ensure_dir,
    safe_remove_tree,
    temporary_directory,
)


class TestEnsureDir:
    """ensure_dir: creates directory and parents, returns resolved path."""

    def test_creates_nested_dir(self, tmp_path_dir):
        sub = tmp_path_dir / "a" / "b" / "c"
        out = ensure_dir(sub)
        assert out.exists()
        assert out.is_dir()
        assert (tmp_path_dir / "a" / "b" / "c").exists()

    def test_idempotent(self, tmp_path_dir):
        sub = tmp_path_dir / "x"
        ensure_dir(sub)
        ensure_dir(sub)
        assert sub.is_dir()


class TestSafeRemoveTree:
    """safe_remove_tree: removes dir; no-op if missing or not a dir."""

    def test_removes_dir(self, tmp_path_dir):
        sub = tmp_path_dir / "d"
        sub.mkdir()
        (sub / "file.txt").write_text("x")
        safe_remove_tree(sub)
        assert not sub.exists()

    def test_no_op_if_not_exists(self, tmp_path_dir):
        safe_remove_tree(tmp_path_dir / "nonexistent")

    def test_no_op_if_file(self, tmp_path_dir):
        f = tmp_path_dir / "file.txt"
        f.write_text("x")
        safe_remove_tree(f)
        assert f.exists()


class TestTemporaryDirectory:
    """temporary_directory: context manager, cleanup on exit."""

    def test_yields_path_and_cleans_up(self):
        with temporary_directory() as d:
            assert d.exists()
            assert d.is_dir()
            (d / "f").write_text("x")
        assert not d.exists()

    def test_prefix_suffix(self):
        with temporary_directory(prefix="myprefix_", suffix="_suffix") as d:
            assert d.name.startswith("myprefix_")
            assert "_suffix" in d.name
