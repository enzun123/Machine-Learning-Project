"""pytest 공통 fixture — OS 무관 pathlib 기준."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """machine-learning-project 루트 (tests/ 의 부모)."""
    return Path(__file__).resolve().parents[1]
