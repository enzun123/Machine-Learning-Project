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

# 본구장·대체구장 정원 (대전 17000: 데이터 상한·실제 관중 규모 반영, 13000은 과소)
STADIUM_ROWS: list[tuple[str, str, int, str]] = [
    ("LG", "잠실", 23750, "서울_잠실"),
    ("두산", "잠실", 23750, "서울_잠실"),
    ("SSG", "인천", 23000, "인천"),
    ("롯데", "사직", 22669, "부산"),
    ("롯데", "울산", 22000, "울산"),
    ("NC", "창원", 22112, "경남_창원"),
    ("NC", "울산", 22000, "울산"),
    ("KIA", "광주", 20500, "광주"),
    ("한화", "대전", 17000, "대전"),
    ("한화", "청주", 10500, "청주"),
    ("삼성", "대구", 24000, "대구"),
    ("삼성", "포항", 9000, "포항"),
    ("KT", "수원", 18700, "경기_수원"),
    ("키움", "고척", 16744, "서울_고척"),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(STADIUM_ROWS, columns=["구단", "구장", "최대수용인원", "region_key"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"행 수: {len(df)} (구장 문자열 기준 조인 — 별칭은 common.stadium_aliases.STADIUM_ALIAS)")


if __name__ == "__main__":
    main()
