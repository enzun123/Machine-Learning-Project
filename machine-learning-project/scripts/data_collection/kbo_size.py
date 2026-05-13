"""
구장별 최대 수용 인원 마스터 생성 (feat/stadium-capacity)

- 수용 인원 정본: data/external/kbo_stadium_info.csv (아래 STADIUM_ROWS를 수정한 뒤 이 스크립트로 갱신)
- 경기 `구장` 별칭 정본: scripts/common/stadium_aliases.py 의 STADIUM_ALIAS (전처리·피처·EDA에서 동일 import)

실행:
  cd machine-learning-project
  python3 scripts/data_collection/kbo_size.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "external" / "kbo_stadium_info.csv"

# KBO 주요 구장 **2026 기준** (사용자 제공: 좌석·명 혼재; 공식 공지 변경 시 수정).
# 창원은 안내 범위(약 17,955석 ~ 22,112명) 중 **최대 수용(명)** 22,112를 사용.
STADIUM_ROWS: list[tuple[str, str, int]] = [
    ("LG", "잠실", 23750),
    ("두산", "잠실", 23750),
    ("SSG", "인천", 23000),
    ("SSG", "문학", 23000),
    ("롯데", "부산", 22669),
    ("롯데", "사직", 22669),
    ("NC", "창원", 22112),
    ("NC", "울산", 22000),
    ("NC", "청주", 10500),
    ("NC", "포항", 9000),
    ("KIA", "광주", 20500),
    ("한화", "대전", 13000),
    ("한화", "한밭", 12000),
    ("삼성", "대구", 29178),
    ("삼성", "포항", 9000),
    ("KT", "수원", 18700),
    ("KT", "청주", 10500),
    ("키움", "고척", 22258),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(STADIUM_ROWS, columns=["구단", "구장", "최대수용인원"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"행 수: {len(df)} (구장 문자열 기준 조인 — 별칭은 common.stadium_aliases.STADIUM_ALIAS)")


if __name__ == "__main__":
    main()
