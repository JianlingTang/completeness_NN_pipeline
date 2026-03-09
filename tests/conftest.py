"""Shared pytest fixtures and path setup for cluster pipeline tests."""
from pathlib import Path

import pytest

# Project root (parent of tests/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


@pytest.fixture
def project_root():
    return ROOT


@pytest.fixture
def tmp_path_dir(tmp_path):
    """Temporary directory for tests that expect a tmp_path_dir fixture (alias for tmp_path)."""
    return tmp_path
