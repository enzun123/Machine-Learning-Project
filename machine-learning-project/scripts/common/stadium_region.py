"""
구장명 → 기상·지역 조인용 `region_key` (단일 소스: data/external/kbo_stadium_info.csv 의 region_key).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
_DEFAULT_STADIUM_CSV = _SCRIPTS.parent / "data" / "external" / "kbo_stadium_info.csv"


@lru_cache(maxsize=1)
def _needles_and_keys(csv_path: str) -> tuple[tuple[str, str], ...]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"구장 정보 CSV 없음: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "구장" not in df.columns or "region_key" not in df.columns:
        raise KeyError("kbo_stadium_info.csv 에 '구장', 'region_key' 컬럼이 필요합니다.")
    pairs: list[tuple[str, str]] = []
    seen_stadium: set[str] = set()
    for _, row in df.iterrows():
        st = str(row["구장"]).strip()
        key = str(row["region_key"]).strip()
        if not st or not key or st in seen_stadium:
            continue
        seen_stadium.add(st)
        pairs.append((st, key))
    pairs.sort(key=lambda x: -len(x[0]))
    return tuple(pairs)


def stadium_to_region_key(stadium: str, *, csv_path: Path | None = None) -> str:
    """구장 문자열에 포함되는 `구장` 부분문자열 중 가장 긴 것으로 region_key 결정."""
    path = csv_path if csv_path is not None else _DEFAULT_STADIUM_CSV
    s = str(stadium).strip()
    for needle, key in _needles_and_keys(str(path.resolve())):
        if needle in s:
            return key
    return "기타"
