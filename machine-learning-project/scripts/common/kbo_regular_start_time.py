"""
KBO 정규시즌 **기본** 개시 시각 (스크랩 표에 시간이 비었을 때 보강용).

규칙 요약 (2026 시즌 운영 안내에 흔히 맞춘 관례; 공식 규정과 다를 수 있음):
- 평일(월~금, 공휴일 아님): 18:30
- 토요일: 17:00 — 7~8월(혹서기)에는 18:00
- 일요일: 14:00 — 7~8월에는 18:00
- 법정·임시 공휴일(평일): 14:00 — 7~8월에는 18:00
- 토요일이 공휴일이면(비혹서기) 토요일 규칙 17:00 유지

공휴일 목록은 연도별로 `KR_PUBLIC_HOLIDAYS_20XX`에 반영합니다. 정부 고시 변경 시 수정하세요.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

# 2026년 법정·대체·일부 임시공휴일 (행정안전부 고시 기준; 변경 시 갱신)
KR_PUBLIC_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),  # 신정
        date(2026, 2, 16),
        date(2026, 2, 17),
        date(2026, 2, 18),  # 설날 연휴
        date(2026, 3, 1),
        date(2026, 3, 2),  # 삼일절·대체
        date(2026, 5, 1),  # 근로자의 날
        date(2026, 5, 5),  # 어린이날
        date(2026, 5, 24),
        date(2026, 5, 25),  # 석가탄신일·대체
        date(2026, 6, 3),  # 지방선거 임시공휴일
        date(2026, 6, 6),  # 현충일(토)
        date(2026, 7, 17),  # 제헌절
        date(2026, 8, 15),
        date(2026, 8, 17),  # 광복절·대체
        date(2026, 9, 24),
        date(2026, 9, 25),
        date(2026, 9, 26),  # 추석 연휴
        date(2026, 10, 3),
        date(2026, 10, 5),  # 개천절 대체
        date(2026, 10, 9),  # 한글날
        date(2026, 12, 25),  # 성탄절
    }
)


def holidays_for_season_year(year: int) -> frozenset[date]:
    if year == 2026:
        return KR_PUBLIC_HOLIDAYS_2026
    return frozenset()


def _strip_parenthesized_weekday(date_cell: str) -> str:
    return re.sub(r"\s*[\(（][^)）]*[\)）]", "", str(date_cell).strip()).strip()


def parse_row_date_only(date_str: str, season_year: int) -> date | None:
    """날짜 셀에서 date만 추출 (시간 무시). 실패 시 None."""
    d = _strip_parenthesized_weekday(date_str)
    parts = [p.strip() for p in re.split(r"[./-]", d) if p.strip()]
    y, month, day = None, None, None
    if len(parts) == 3:
        p0 = parts[0]
        month, day = int(parts[1]), int(parts[2])
        if p0.isdigit() and len(p0) == 2:
            y = int(p0) + 2000
            if abs(y - season_year) > 1:
                y = season_year
        else:
            y = int(p0)
            if y < 100:
                y += 2000
    elif len(parts) == 2:
        month, day = int(parts[0]), int(parts[1])
        y = season_year
    else:
        return None
    try:
        return date(y, month, day)
    except (ValueError, TypeError):
        return None


def default_start_hm(game_d: date, season_year: int) -> tuple[int, int]:
    """(시, 분) 기본 개시 시각."""
    hol = holidays_for_season_year(season_year)
    wd = game_d.weekday()
    month = game_d.month
    heat = month in (7, 8) and season_year >= 2026
    is_hol = game_d in hol

    if heat:
        if wd >= 5 or is_hol:
            return 18, 0
        return 18, 30

    if wd == 5:
        return 17, 0
    if wd == 6:
        return 14, 0
    if is_hol:
        return 14, 0
    return 18, 30


def _time_cell_has_clock(s: object) -> bool:
    t = str(s).strip()
    if not t:
        return False
    return bool(re.search(r"\d{1,2}\s*:\s*\d{2}", t))


def apply_default_start_time_strings(
    date_col: pd.Series,
    time_col: pd.Series,
    season_year: int,
    *,
    from_year: int = 2026,
) -> pd.Series:
    """
    `경기시간`이 비었거나 시:분 형태가 아니면 기본 시각 문자열로 채움.
    `season_year < from_year` 이면 원본 유지.
    """
    if season_year < from_year:
        return time_col.astype(str)

    out: list[str] = []
    for dcell, tcell in zip(date_col, time_col, strict=False):
        if _time_cell_has_clock(tcell):
            out.append(str(tcell).strip())
            continue
        gd = parse_row_date_only(str(dcell), season_year)
        if gd is None:
            out.append(str(tcell).strip() if tcell is not None else "")
            continue
        h, m = default_start_hm(gd, season_year)
        out.append(f"{h:d}:{m:02d}")
    return pd.Series(out, index=date_col.index, dtype=str)

