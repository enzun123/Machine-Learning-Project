"""common 모듈 스모크 테스트 — 네트워크·Chrome 불필요 (macOS / Windows 공통)."""

from __future__ import annotations

from datetime import date

import pytest

from common.congestion_levels import classify_congestion_pct
from common.kbo_regular_start_time import default_start_hm, parse_row_date_only
from common.kma_vilage_fcst import redact_api_secrets, rule_band_from_mm_h, rule_band_from_pop
from common.stadium_aliases import (
    HOME_STADIUM_BY_TEAM,
    is_secondary_stadium,
    stadium_for_model_ohe,
)
from common.stadium_capacity import load_stadium_capacity_map, venue_clip_capacity


@pytest.mark.parametrize(
    ("pct", "level"),
    [(30.0, "LOW"), (65.0, "NORMAL"), (90.0, "HIGH")],
)
def test_classify_congestion_pct(pct: float, level: str) -> None:
    plan = classify_congestion_pct(pct)
    assert plan.level == level


def test_secondary_stadium_and_ohe() -> None:
    assert is_secondary_stadium("울산") is True
    assert is_secondary_stadium("잠실") is False
    assert stadium_for_model_ohe("울산", "롯데") == HOME_STADIUM_BY_TEAM["롯데"]


@pytest.mark.parametrize(
    ("date_str", "expected"),
    [
        ("2025.07.15", date(2025, 7, 15)),
        ("07-15", date(2025, 7, 15)),
    ],
)
def test_parse_row_date_only(date_str: str, expected: date) -> None:
    assert parse_row_date_only(date_str, season_year=2025) == expected


def test_default_start_hm_weekday_evening() -> None:
    hm = default_start_hm(date(2025, 4, 15), season_year=2025)
    assert hm == (18, 30)


def test_redact_api_secrets() -> None:
    raw = "https://api.example?authKey=SECRET&serviceKey=OTHER"
    out = redact_api_secrets(raw)
    assert "SECRET" not in out
    assert "OTHER" not in out
    assert "***" in out


@pytest.mark.parametrize(
    ("mm", "css"),
    [(0.0, "low"), (7.0, "mid"), (12.0, "high")],
)
def test_rule_band_from_mm_h(mm: float, css: str) -> None:
    _title, _body, band = rule_band_from_mm_h(mm)
    assert band == css


def test_rule_band_from_pop_high() -> None:
    _title, _body, band = rule_band_from_pop(80.0)
    assert band == "high"


def test_venue_clip_capacity_secondary_fallback(project_root) -> None:
    cap_map = load_stadium_capacity_map(project_root)
    if not cap_map:
        pytest.skip("kbo_stadium_info.csv 없음")
    cap = venue_clip_capacity("울산", "롯데", cap_map)
    assert cap > 0
