"""추론 피처: game_date 이전 필터·wind_bucket 재계산."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from modeling.batch_feature_builder import (
    build_one_feature_row,
    filter_train_before_date,
    _wind_bucket,
)
from modeling.train_model import FEATURE_COLUMNS, load_training_table


@pytest.fixture(scope="module")
def train_ready(project_root: Path) -> pd.DataFrame:
    p = project_root / "data" / "processed" / "kbo_train_ready.csv"
    if not p.is_file():
        pytest.skip("kbo_train_ready.csv 없음")
    return load_training_table(p)


def test_filter_train_before_date_excludes_future_week(train_ready: pd.DataFrame) -> None:
    cut = pd.Timestamp("2099-12-31")
    sub = filter_train_before_date(train_ready, cut)
    assert len(sub) == len(train_ready)
    early = pd.Timestamp("2024-04-01")
    sub2 = filter_train_before_date(train_ready, early)
    assert len(sub2) < len(train_ready)
    assert int(sub2["연도"].max()) <= 2024
    if int(sub2["연도"].max()) == 2024:
        assert int(sub2["월"].max()) <= 4


def test_wind_bucket_matches_training_bins() -> None:
    assert _wind_bucket(0.5) == "Calm"
    assert _wind_bucket(2.0) == "Light"
    assert _wind_bucket(4.0) == "Moderate"
    assert _wind_bucket(6.0) == "Strong"


def test_build_row_sets_wind_bucket_from_input(train_ready: pd.DataFrame) -> None:
    cap = {str(g): 20000 for g in train_ready["구장"].dropna().unique()}
    row = build_one_feature_row(
        train_ready,
        home="KIA",
        away="LG",
        stadium="광주",
        game_date=pd.Timestamp("2025-10-01"),
        temp_c=20.0,
        rain_mm=0.0,
        hum_pct=55.0,
        cap_map=cap,
        wind_mps=6.0,
    )
    assert row["wind_bucket"] == "Strong"
    assert set(row.keys()) >= set(FEATURE_COLUMNS)
