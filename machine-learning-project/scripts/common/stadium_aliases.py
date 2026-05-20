"""
경기 데이터 `구장` 문자열을 `kbo_stadium_info.csv` 조인에 맞게 표준화할 때 쓰는 별칭.

- 최대 수용 인원·구단별 구장 행 정본: `data/external/kbo_stadium_info.csv`
  (`scripts/data_collection/kbo_size.py`의 STADIUM_ROWS → 위 CSV로 기록).
- 수치·행 추가/수정은 `kbo_size.py`만 고치고 이 스크립트를 다시 실행하면 된다.
- 원천에만 나오는 구장 표기 → 표준 `구장`명은 `STADIUM_ALIAS`가 정본이다.
"""

from __future__ import annotations

STADIUM_ALIAS: dict[str, str] = {
    "한밭": "대전",
    "문학": "인천",
    "부산": "사직",
}

# 2군·대체 구장 → 홈팀 본구장 (OHE `구장` 통합용). 정원은 실제 구장 기준 유지.
SECONDARY_STADIUM_NAMES: frozenset[str] = frozenset({"울산", "청주", "포항"})

HOME_STADIUM_BY_TEAM: dict[str, str] = {
    "LG": "잠실",
    "두산": "잠실",
    "SSG": "인천",
    "KT": "수원",
    "키움": "고척",
    "KIA": "광주",
    "한화": "대전",
    "삼성": "대구",
    "NC": "창원",
    "롯데": "사직",
}


def is_secondary_stadium(stadium: str) -> bool:
    return str(stadium).strip() in SECONDARY_STADIUM_NAMES


def stadium_for_model_ohe(stadium: str, home_team: str) -> str:
    """범주형 `구장`·stadium_x_rain: 대체 구장이면 홈팀 본구장."""
    st = str(stadium).strip()
    if is_secondary_stadium(st):
        return HOME_STADIUM_BY_TEAM.get(str(home_team).strip(), st)
    return st


def is_small_stadium_game(actual_stadium: str) -> bool:
    """소형·대체 구장 여부(정원 1.5만 미만 본구장과 구분)."""
    return is_secondary_stadium(actual_stadium)
