"""
대체 구장(포항·울산·청주) 실제 개최지 관중 이력 — final_dataset 기준.
학습·예측 시 home_last5·matchup 등이 본구장(대구) 수준으로 끌려 올라가지 않도록 보정.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common.stadium_aliases import STADIUM_ALIAS, is_secondary_stadium

_ATTENDANCE_SCALE_FEATURES = (
    "home_prior_mean_att",
    "visitor_prior_mean_att",
    "home_last5_mean_att",
    "visitor_last5_mean_att",
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
    """해당 홈·실제 구장(및 매치업·원정)의 과거 관중 요약."""
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
    venue_mean: float | None = None
    if len(sub) >= 1:
        att = pd.to_numeric(sub["관중수"], errors="coerce").dropna()
        if len(att) >= 1:
            venue_mean = float(att.mean())
            out["home_prior_mean_att"] = venue_mean
            out["home_last5_mean_att"] = float(att.tail(5).mean())

    mu = df[
        (df["홈팀"].astype(str) == home_s)
        & (df["방문팀"].astype(str) == away_s)
        & (df["구장"].astype(str) == st)
    ]
    if before is not None and "경기날짜" in mu.columns:
        mu = mu[mu["경기날짜"] < pd.Timestamp(before)]
    mu_att = pd.to_numeric(mu["관중수"], errors="coerce").dropna()
    if len(mu_att) >= 1:
        out["matchup_prior_mean_att"] = float(mu_att.mean())
    elif venue_mean is not None:
        out["matchup_prior_mean_att"] = venue_mean

    vis = df[(df["방문팀"].astype(str) == away_s) & (df["구장"].astype(str) == st)]
    if before is not None and "경기날짜" in vis.columns:
        vis = vis[vis["경기날짜"] < pd.Timestamp(before)]
    vis_att = pd.to_numeric(vis["관중수"], errors="coerce").dropna()
    if len(vis_att) >= 1:
        out["visitor_prior_mean_att"] = float(vis_att.mean())
        out["visitor_last5_mean_att"] = float(vis_att.tail(5).mean())
    elif venue_mean is not None:
        out["visitor_prior_mean_att"] = venue_mean
        out["visitor_last5_mean_att"] = out.get("home_last5_mean_att", venue_mean)

    return out


def apply_secondary_priors_to_row(
    row: dict,
    root: Path,
    home: str,
    away: str,
    actual_stadium: str,
    game_date: pd.Timestamp | None,
) -> dict:
    """피처 dict — 대체 구장 실제 관중 수준으로 덮어쓰기·상한."""
    st = STADIUM_ALIAS.get(str(actual_stadium).strip(), str(actual_stadium).strip())
    if not is_secondary_stadium(st):
        return row

    before = pd.Timestamp(game_date) if game_date is not None else None
    priors = venue_attendance_priors(root, home, away, st, before=before)
    if not priors:
        return row

    venue_mean = priors.get("home_prior_mean_att") or priors.get("home_last5_mean_att")
    if venue_mean is None or not np.isfinite(venue_mean):
        return row

    out = dict(row)
    cap_hi = float(venue_mean) * 1.12

    for k in _ATTENDANCE_SCALE_FEATURES:
        if k in priors and np.isfinite(priors[k]):
            out[k] = float(priors[k])
        elif k in out:
            try:
                v = float(out[k])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > cap_hi:
                out[k] = cap_hi

    return out
