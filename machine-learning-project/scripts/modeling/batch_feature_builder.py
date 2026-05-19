"""
일정 CSV(날짜·팀·구장) → 모델 입력 피처 행 생성 (Streamlit 단일 예측과 동일 로직).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.stadium_aliases import (
    HOME_STADIUM_BY_TEAM,
    STADIUM_ALIAS,
    is_secondary_stadium,
    is_small_stadium_game,
    stadium_for_model_ohe,
)
from modeling.train_model import FEATURE_COLUMNS

RAIN_LABELS_ML = ["No_Rain", "Rain_0_1mm", "Rain_1_5mm", "Rain_5mm_plus"]
_DERBY_PAIRS = {
    frozenset({"LG", "두산"}),
    frozenset({"롯데", "NC"}),
    frozenset({"삼성", "KIA"}),
    frozenset({"KT", "키움"}),
    frozenset({"SSG", "한화"}),
}

_SCHEDULE_COLUMN_ALIASES: dict[str, list[str]] = {
    "경기날짜": ["경기날짜", "날짜", "date", "Date", "일자"],
    "홈팀": ["홈팀", "home", "홈", "Home"],
    "방문팀": ["방문팀", "원정", "원정팀", "away", "Away"],
    "구장": ["구장", "stadium", "경기장", "Stadium"],
    "기온": ["일평균기온(°C)", "일평균기온", "기온", "temp", "temperature"],
    "강수": ["일합계강수량(mm)", "일합계강수량", "강수", "rain", "rainfall_mm"],
    "습도": ["일평균상대습도(%)", "일평균상대습도", "습도", "hum", "humidity"],
    "관중수": ["관중수", "관중", "attendance"],
}


def load_stadium_capacity_map(root: Path) -> dict[str, int]:
    p = root / "data" / "external" / "kbo_stadium_info.csv"
    if not p.is_file():
        return {}
    st_df = pd.read_csv(p, encoding="utf-8-sig")
    cap: dict[str, int] = {}
    for _, r in st_df.iterrows():
        gu = STADIUM_ALIAS.get(str(r["구장"]).strip(), str(r["구장"]).strip())
        cap[gu] = int(r["최대수용인원"])
    return cap


def normalize_schedule_csv(df: pd.DataFrame) -> pd.DataFrame:
    """다양한 헤더명 → 경기날짜, 홈팀, 방문팀, 구장 (+ 선택 기상·관중)."""
    col_map: dict[str, str] = {}
    for col in df.columns:
        c = str(col).strip()
        for canonical, aliases in _SCHEDULE_COLUMN_ALIASES.items():
            if c == canonical or c.lower() == canonical.lower():
                col_map[col] = canonical
                break
            if any(c == a or c.lower() == a.lower() for a in aliases):
                col_map[col] = canonical
                break
    out = df.rename(columns=col_map)
    required = ["경기날짜", "홈팀", "방문팀", "구장"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(
            f"일정 CSV 필수 컬럼 누락: {missing}. "
            f"필요: 날짜·홈팀·방문팀·구장 (예: 경기날짜, 홈팀, 방문팀, 구장)"
        )
    return out


def _rain_bucket(mm: float) -> str:
    x = float(mm)
    if -1.0 <= x <= 0.0:
        return RAIN_LABELS_ML[0]
    if 0.0 < x <= 1.0:
        return RAIN_LABELS_ML[1]
    if 1.0 < x <= 5.0:
        return RAIN_LABELS_ML[2]
    return RAIN_LABELS_ML[3]


def _temp_bucket(t: float) -> str:
    s = pd.cut(
        pd.Series([float(t)]),
        bins=[-float("inf"), 10, 20, 25, 30, float("inf")],
        labels=["VeryCold", "Cold", "Mild", "Warm", "Hot"],
    )
    v = s.iloc[0]
    return "Mild" if pd.isna(v) else str(v)


def _hum_bucket(h: float) -> str:
    s = pd.cut(
        pd.Series([float(h)]),
        bins=[0, 40, 60, 80, 100],
        labels=["Dry", "Normal", "Humid", "VeryHumid"],
    )
    v = s.iloc[0]
    return "Normal" if pd.isna(v) else str(v)


def _pick_template(tr: pd.DataFrame, home: str, away: str, stadium: str) -> pd.Series:
    g = tr.copy()
    g["_g"] = g["구장"].astype(str).replace(STADIUM_ALIAS)
    st_n = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    st_ohe = stadium_for_model_ohe(st_n, home)
    hs, vs = str(home), str(away)
    for m in [
        (g["홈팀"].astype(str) == hs) & (g["방문팀"].astype(str) == vs) & (g["_g"] == st_ohe),
        (g["홈팀"].astype(str) == hs) & (g["_g"] == st_ohe),
        g["_g"] == st_ohe,
    ]:
        sub = g.loc[m]
        if len(sub) >= 1:
            return sub.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]
    return g.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]


def _get_cap(stadium: str, home: str, cap_map: dict[str, int]) -> int:
    st = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    if st in cap_map:
        return int(cap_map[st])
    if is_secondary_stadium(st):
        main = HOME_STADIUM_BY_TEAM.get(str(home).strip(), st)
        return int(cap_map.get(main, 20_000))
    return 20_000


def build_one_feature_row(
    tr: pd.DataFrame,
    *,
    home: str,
    away: str,
    stadium: str,
    game_date: pd.Timestamp,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
    cap_map: dict[str, int],
) -> dict:
    row = _pick_template(tr, home, away, stadium)
    gdt = pd.Timestamp(game_date)
    wdn = int(gdt.dayofweek)
    st_actual = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    st_key = stadium_for_model_ohe(st_actual, home)
    if is_secondary_stadium(st_actual):
        main_st = HOME_STADIUM_BY_TEAM.get(str(home).strip(), st_key)
        ml_cap = float(cap_map.get(main_st, _get_cap(stadium, home, cap_map)))
    else:
        ml_cap = float(_get_cap(stadium, home, cap_map))
    is_rain_i = int(float(rain_mm) > 0)

    d = {c: row.get(c, np.nan) for c in FEATURE_COLUMNS}
    upd = {
        "연도": int(gdt.year),
        "월": int(gdt.month),
        "주차_ISO": int(gdt.isocalendar().week),
        "홈팀": str(home),
        "방문팀": str(away),
        "구장": st_key,
        "stadium_capacity": ml_cap,
        "is_capacity_missing": 0,
        "is_rain": is_rain_i,
        "rain_bucket": _rain_bucket(rain_mm),
        "temp_bucket": _temp_bucket(temp_c),
        "is_hot": int(float(temp_c) >= 30),
        "humidity_bucket": _hum_bucket(hum_pct),
        "is_weekend": int(wdn >= 5),
        "is_friday": int(wdn == 4),
        "is_saturday": int(wdn == 5),
        "is_sunday": int(wdn == 6),
        "weekday_sin": float(np.sin(2 * np.pi * wdn / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * wdn / 7)),
        "stadium_x_rain": f"{st_key}_{is_rain_i}",
        "is_small_stadium": int(is_small_stadium_game(st_actual)),
        "is_derby": int(frozenset({str(home), str(away)}) in _DERBY_PAIRS),
        "is_childrens_day": int(gdt.month == 5 and gdt.day in (4, 5, 6)),
    }
    for k, v in upd.items():
        if k in d:
            d[k] = v
    return d


def build_features_from_schedule(
    schedule: pd.DataFrame,
    train_ready: pd.DataFrame,
    cap_map: dict[str, int],
    *,
    default_temp: float = 18.0,
    default_rain: float = 0.0,
    default_hum: float = 55.0,
) -> pd.DataFrame:
    """일정 표 → FEATURE_COLUMNS DataFrame."""
    sched = normalize_schedule_csv(schedule)
    rows: list[dict] = []
    for _, r in sched.iterrows():
        dt = pd.to_datetime(r["경기날짜"], errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"날짜 파싱 실패: {r['경기날짜']}")
        temp = float(r["기온"]) if "기온" in sched.columns and pd.notna(r.get("기온")) else default_temp
        rain = float(r["강수"]) if "강수" in sched.columns and pd.notna(r.get("강수")) else default_rain
        hum = float(r["습도"]) if "습도" in sched.columns and pd.notna(r.get("습도")) else default_hum
        rows.append(
            build_one_feature_row(
                train_ready,
                home=str(r["홈팀"]).strip(),
                away=str(r["방문팀"]).strip(),
                stadium=str(r["구장"]).strip(),
                game_date=dt,
                temp_c=temp,
                rain_mm=rain,
                hum_pct=hum,
                cap_map=cap_map,
            )
        )
    feat = pd.DataFrame(rows)
    if "관중수" in sched.columns:
        feat["관중수"] = pd.to_numeric(sched["관중수"], errors="coerce")
    return feat


def schedule_template_path(root: Path) -> Path:
    return root / "data" / "external" / "batch_predict_schedule_template.csv"
