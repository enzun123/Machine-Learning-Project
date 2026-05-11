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
}
