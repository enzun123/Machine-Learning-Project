"""커밋된 데이터·학습 테이블 스모크 — 경로는 pathlib (macOS / Windows 공통)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modeling.train_model import TARGET, load_training_table


@pytest.fixture(scope="module")
def train_ready_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / "kbo_train_ready.csv"


def test_committed_artifacts_exist(project_root: Path) -> None:
    required = [
        project_root / "data" / "processed" / "kbo_train_ready.csv",
        project_root / "data" / "external" / "kbo_stadium_info.csv",
        project_root / "models" / "attendance_rf_pipeline.joblib",
        project_root / "models" / "train_report.json",
    ]
    missing = [p for p in required if not p.is_file()]
    assert not missing, f"누락 파일: {missing}"


def test_load_training_table_smoke(train_ready_path: Path) -> None:
    if not train_ready_path.is_file():
        pytest.skip("kbo_train_ready.csv 없음")
    df = load_training_table(train_ready_path)
    assert len(df) > 0
    assert TARGET in df.columns
    assert df[TARGET].notna().all()


def test_train_report_has_metrics(project_root: Path) -> None:
    report_path = project_root / "models" / "train_report.json"
    if not report_path.is_file():
        pytest.skip("train_report.json 없음")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"]["mae"] < report["baseline_dummy_mean"]["mae"]
