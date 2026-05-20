"""
대체 구장(포항·울산·청주) 실제 개최지 관중 이력 — final_dataset 기준.
학습·예측 시 home_last5 등을 본구장(대구)이 아닌 실제 구장 수준으로 맞춤.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common.stadium_aliases import STADIUM_ALIAS, is_secondary_stadium

PRIOR_KEYS = (
    "home_prior_mean_att",
    "home_last5_mean_att",
    "matchup_prior_mean_att",
)


@lru_cache(maxsize=1)
def _load_final_attendance(root_str: str) -> pd.DataFrame:
    p = Path(root_str) / "data" / "processed" / "final_dataset.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "경기날짜" in df.columns:
        df["경기날짜"] = pd.to_datetime(df["경기날짜"], errors="coerce")
    return df


def venue_attendance_priors(
    root: Path,
    home: str,
    away: str,
    actual_stadium: str,
    *,
    before: pd.Timestamp | None = None,
) -> dict[str, float]:
    """해당 홈·실제 구장(및 매치업)의 과거 관중 요약."""
    st = STADIUM_ALIAS.get(str(actual_stadium).strip(), str(actual_stadium).strip())
    if not is_secondary_stadium(st):
        return {}

    df = _load_final_attendance(str(root.resolve()))
    if df.empty or "관중수" not in df.columns:
        return {}

    home_s = str(home).strip()
    away_s = str(away).strip()
    sub = df[(df["홈팀"].astype(str) == home_s) & (df["구장"].astype(str) == st)].copy()
    if before is not None and "경기날짜" in sub.columns:
        sub = sub[sub["경기날짜"] < pd.Timestamp(before)]

    out: dict[str, float] = {}
    if len(sub) >= 1:
        att = pd.to_numeric(sub["관중수"], errors="coerce").dropna()
        if len(att) >= 1:
            out["home_prior_mean_att"] = float(att.mean())
            out["home_last5_mean_att"] = float(att.tail(5).mean())

    mu = df[
        (df["홈팀"].astype(str) == home_s) & (df["방문팀"].astype(str) == away_s) & (df["구장"].astype(str) == st)
    ]
    if before is not None and "경기날짜" in mu.columns:
        mu = mu[mu["경기날짜"] < pd.Timestamp(before)]
    mu_att = pd.to_numeric(mu["관중수"], errors="coerce").dropna()
    if len(mu_att) >= 1:
        out["matchup_prior_mean_att"] = float(mu_att.mean())

    return out


def apply_secondary_priors_to_row(
    row: dict,
    root: Path,
    home: str,
    away: str,
    actual_stadium: str,
    game_date: pd.Timestamp | None,
) -> dict:
    """피처 dict에 대체 구장 관중 prior 덮어쓰기."""
    st = STADIUM_ALIAS.get(str(actual_stadium).strip(), str(actual_stadium).strip())
    if not is_secondary_stadium(st):
        return row
    before = pd.Timestamp(game_date) if game_date is not None else None
    priors = venue_attendance_priors(root, home, away, st, before=before)
    out = dict(row)
    for k, v in priors.items():
        if k in out and np.isfinite(v):
            out[k] = float(v)
    return out
