"""
구장별 최대 수용 인원 — kbo_stadium_info.csv + (선택) kbo_train_ready 피처 병합.
UI 예측 상한·혼잡도·batch 클리핑 공용.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.stadium_aliases import STADIUM_ALIAS, HOME_STADIUM_BY_TEAM, is_secondary_stadium


def load_stadium_capacity_map(root: Path) -> dict[str, int]:
    """kbo_stadium_info.csv → {구장명: 최대수용인원} (별칭·원본 이름 모두 등록)."""
    p = root / "data" / "external" / "kbo_stadium_info.csv"
    if not p.is_file():
        return {}
    st_df = pd.read_csv(p, encoding="utf-8-sig")
    if not {"구장", "최대수용인원"}.issubset(st_df.columns):
        return {}
    cap: dict[str, int] = {}
    for _, r in st_df.iterrows():
        gu = str(r["구장"]).strip()
        n = int(pd.to_numeric(r["최대수용인원"], errors="coerce"))
        cap[gu] = n
        aliased = STADIUM_ALIAS.get(gu, gu)
        cap[aliased] = n
    return cap


def merge_train_ready_capacities(cap_map: dict[str, int], root: Path) -> dict[str, int]:
    """학습 피처의 stadium_capacity(본구장 OHE 기준)로 본구장 정원 갱신."""
    p = root / "data" / "processed" / "kbo_train_ready.csv"
    if not p.is_file():
        return cap_map
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "구장" not in df.columns or "stadium_capacity" not in df.columns:
        return cap_map
    out = dict(cap_map)
    g = (
        df.dropna(subset=["구장", "stadium_capacity"])
        .groupby(df["구장"].astype(str).str.strip())["stadium_capacity"]
        .max()
    )
    for gu, val in g.items():
        if pd.notna(val):
            out[str(gu)] = int(val)
    return out


def load_capacity_map_for_app(root: Path) -> dict[str, int]:
    """앱·예측용: CSV 마스터 + 학습 테이블 병합."""
    cap = load_stadium_capacity_map(root)
    return merge_train_ready_capacities(cap, root)


def venue_clip_capacity(stadium: str, home: str, cap_map: dict[str, int]) -> int:
    """예측 상한·혼잡도용 — 실제 경기 구장 정원 (포항·울산 등 대체 구장 포함)."""
    raw = str(stadium).strip()
    st = STADIUM_ALIAS.get(raw, raw)
    if st in cap_map:
        return int(cap_map[st])
    if raw in cap_map:
        return int(cap_map[raw])
    if is_secondary_stadium(st):
        main = HOME_STADIUM_BY_TEAM.get(str(home).strip(), st)
        if main in cap_map:
            return int(cap_map[main])
    return 20_000


def model_feature_capacity(stadium: str, home: str, cap_map: dict[str, int]) -> int:
    """ML 피처 stadium_capacity — 실제 개최 구장 정원(포항·울산·청주 포함)."""
    return venue_clip_capacity(stadium, home, cap_map)
