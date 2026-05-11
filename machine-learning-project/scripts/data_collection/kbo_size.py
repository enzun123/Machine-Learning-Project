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

# KBO 공개 자료·구장 안내 기준(연도에 따라 변동 가능). 수치 수정 시 이 파일만 갱신하면 됨.
STADIUM_ROWS: list[tuple[str, str, int]] = [
    ("LG", "잠실", 25000),
    ("두산", "잠실", 25000),
    ("SSG", "인천", 23000),
    ("SSG", "문학", 23000),
    ("롯데", "부산", 27500),
    ("롯데", "사직", 27500),
    ("NC", "창원", 22000),
    ("NC", "울산", 22000),
    ("NC", "청주", 10500),
    ("NC", "포항", 9000),
    ("KIA", "광주", 20500),
    ("한화", "대전", 20000),
    ("한화", "한밭", 12000),
    ("삼성", "대구", 24000),
    ("삼성", "포항", 9000),
    ("KT", "수원", 20000),
    ("KT", "청주", 10500),
    ("키움", "고척", 16500),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(STADIUM_ROWS, columns=["구단", "구장", "최대수용인원"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"행 수: {len(df)} (구장 문자열 기준 조인 — 별칭은 common.stadium_aliases.STADIUM_ALIAS)")


if __name__ == "__main__":
    main()
