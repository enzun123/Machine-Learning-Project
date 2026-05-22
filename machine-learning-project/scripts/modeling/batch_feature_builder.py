"""
일정 CSV(날짜·팀·구장) → 모델 입력 피처 행 생성 (Streamlit 단일 예측과 동일 로직).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.stadium_aliases import (
    STADIUM_ALIAS,
    is_secondary_stadium,
    is_small_stadium_game,
    stadium_for_model_ohe,
)
from common.secondary_venue_stats import apply_secondary_priors_to_row
from common.stadium_capacity import (
    load_capacity_map_for_app,
    model_feature_capacity,
    venue_clip_capacity,
)
from common.config import (
    DEFAULT_WIND_MEDIAN_FALLBACK,
    WIND_BUCKET_EDGES,
    WIND_BUCKET_LABELS,
)
from modeling.train_model import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    "풍속": ["일평균풍속(m/s)", "일평균풍속", "풍속", "wind", "wind_mps"],
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


def _detect_csv_sep(sample: str) -> str:
    line = sample.split("\n", 1)[0] if sample else ""
    if line.count(";") >= max(1, line.count(",")) and ";" in line:
        return ";"
    if "\t" in line and line.count("\t") >= line.count(","):
        return "\t"
    return ","


def read_schedule_csv_bytes(data: bytes) -> pd.DataFrame:
    """업로드 바이트 → 일정 DataFrame (인코딩·구분자 자동)."""
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        sep = _detect_csv_sep(text[:4096])
        df = pd.read_csv(
            pd.io.common.StringIO(text),
            sep=sep,
            encoding=enc,
            skipinitialspace=True,
        )
        try:
            return normalize_schedule_csv(df)
        except KeyError as e:
            last_err = e
            if sep != ",":
                df = pd.read_csv(pd.io.common.StringIO(text), sep=",", encoding=enc)
                try:
                    return normalize_schedule_csv(df)
                except KeyError as e2:
                    last_err = e2
    if last_err is not None:
        raise last_err
    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")


def _is_blank_cell(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    return s == "" or s.lower() in ("nan", "nat", "none", "<na>")


def parse_schedule_date(value) -> pd.Timestamp | None:
    if _is_blank_cell(value):
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
        try:
            ts = pd.to_datetime(float(value), unit="D", origin="1899-12-30", errors="coerce")
            if pd.notna(ts):
                return pd.Timestamp(ts).normalize()
        except (TypeError, ValueError, OverflowError):
            pass
    s = str(value).strip()
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return pd.Timestamp(dt).normalize()
    if "." in s:
        parts = [p.strip() for p in s.split(".") if p.strip()]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
            return pd.Timestamp(year=y, month=m, day=d)
    return None


def normalize_schedule_csv(df: pd.DataFrame) -> pd.DataFrame:
    """다양한 헤더명 → 경기날짜, 홈팀, 방문팀, 구장 (+ 선택 기상·관중)."""
    col_map: dict[str, str] = {}
    for col in df.columns:
        c = str(col).strip().lstrip("\ufeff")
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
            f"현재 열: {list(df.columns)[:12]}{'…' if len(df.columns) > 12 else ''}. "
            f"엑셀은 'CSV UTF-8'로 저장하거나, 구분 기호가 ; 이면 세미콜론 CSV로 저장해 주세요."
        )
    mask = out[required].apply(
        lambda row: any(not _is_blank_cell(v) for v in row),
        axis=1,
    )
    out = out.loc[mask].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("유효한 경기 행이 없습니다. 경기날짜·홈팀·방문팀·구장을 채워 주세요.")
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


def _wind_bucket(wind_mps: float) -> str:
    s = pd.cut(
        pd.Series([float(wind_mps)]),
        bins=list(WIND_BUCKET_EDGES),
        labels=list(WIND_BUCKET_LABELS),
    )
    v = s.iloc[0]
    return "Calm" if pd.isna(v) else str(v)


def filter_train_before_date(tr: pd.DataFrame, game_date) -> pd.DataFrame:
    """추론 시점(game_date) 이전 경기만 사용 — 폼·prior 피처 시점 누수 방지."""
    gdt = pd.Timestamp(game_date).normalize()
    if "경기날짜" in tr.columns:
        dates = pd.to_datetime(tr["경기날짜"], errors="coerce")
        return tr.loc[dates < gdt].copy()
    if {"연도", "월", "주차_ISO"}.issubset(tr.columns):
        y = int(gdt.year)
        m = int(gdt.month)
        w = int(gdt.isocalendar().week)
        yr = pd.to_numeric(tr["연도"], errors="coerce")
        mo = pd.to_numeric(tr["월"], errors="coerce")
        wk = pd.to_numeric(tr["주차_ISO"], errors="coerce")
        before = (
            (yr < y)
            | ((yr == y) & (mo < m))
            | ((yr == y) & (mo == m) & (wk < w))
        )
        return tr.loc[before.fillna(False)].copy()
    return tr.copy()


def _pick_template(tr: pd.DataFrame, home: str, away: str, stadium: str) -> pd.Series:
    g = tr.copy()
    g["_g"] = g["구장"].astype(str).replace(STADIUM_ALIAS)
    st_n = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    st_ohe = stadium_for_model_ohe(st_n, home)
    hs, vs = str(home), str(away)

    if is_secondary_stadium(st_n) and "is_small_stadium" in g.columns:
        small_home = g[(g["홈팀"].astype(str) == hs) & (g["is_small_stadium"] == 1)]
        if len(small_home) >= 1:
            return small_home.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]

    for m in [
        (g["홈팀"].astype(str) == hs) & (g["방문팀"].astype(str) == vs) & (g["_g"] == st_ohe),
        (g["홈팀"].astype(str) == hs) & (g["_g"] == st_ohe),
        g["_g"] == st_ohe,
    ]:
        sub = g.loc[m]
        if len(sub) >= 1:
            return sub.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]
    return g.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]


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
    wind_mps: float | None = None,
) -> dict:
    tr_hist = filter_train_before_date(tr, game_date)
    if len(tr_hist) == 0:
        tr_hist = tr
    row = _pick_template(tr_hist, home, away, stadium)
    gdt = pd.Timestamp(game_date)
    wdn = int(gdt.dayofweek)
    st_actual = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    st_key = stadium_for_model_ohe(st_actual, home)
    ml_cap = float(model_feature_capacity(stadium, home, cap_map))
    is_rain_i = int(float(rain_mm) > 0)
    wind = float(wind_mps) if wind_mps is not None else float(DEFAULT_WIND_MEDIAN_FALLBACK)

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
        "wind_bucket": _wind_bucket(wind),
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
    d = apply_secondary_priors_to_row(
        d,
        PROJECT_ROOT,
        home,
        away,
        st_actual,
        gdt,
    )
    return d


def build_ml_feature_dataframe(
    tr: pd.DataFrame,
    *,
    home: str,
    away: str,
    stadium: str,
    game_date,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
    cap_map: dict[str, int],
    wind_mps: float | None = None,
) -> pd.DataFrame:
    """단일 경기 ML 입력 (Streamlit·CLI 공용)."""
    d = build_one_feature_row(
        tr,
        home=home,
        away=away,
        stadium=stadium,
        game_date=pd.Timestamp(game_date),
        temp_c=temp_c,
        rain_mm=rain_mm,
        hum_pct=hum_pct,
        cap_map=cap_map,
        wind_mps=wind_mps,
    )
    return pd.DataFrame([d])[FEATURE_COLUMNS]


def build_features_from_schedule(
    schedule: pd.DataFrame,
    train_ready: pd.DataFrame,
    cap_map: dict[str, int],
    *,
    default_temp: float = 18.0,
    default_rain: float = 0.0,
    default_hum: float = 55.0,
    default_wind: float = DEFAULT_WIND_MEDIAN_FALLBACK,
) -> pd.DataFrame:
    """일정 표 → FEATURE_COLUMNS DataFrame."""
    sched = normalize_schedule_csv(schedule)
    rows: list[dict] = []
    bad_dates: list[str] = []
    for i, r in sched.iterrows():
        dt = parse_schedule_date(r["경기날짜"])
        if dt is None:
            bad_dates.append(f"{int(i) + 2}행: {r['경기날짜']!r}")
            continue
        temp = float(r["기온"]) if "기온" in sched.columns and pd.notna(r.get("기온")) else default_temp
        rain = float(r["강수"]) if "강수" in sched.columns and pd.notna(r.get("강수")) else default_rain
        hum = float(r["습도"]) if "습도" in sched.columns and pd.notna(r.get("습도")) else default_hum
        wind = (
            float(r["풍속"])
            if "풍속" in sched.columns and pd.notna(r.get("풍속"))
            else default_wind
        )
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
                wind_mps=wind,
            )
        )
    if bad_dates:
        raise ValueError("날짜 파싱 실패:\n" + "\n".join(bad_dates[:8]))
    if not rows:
        raise ValueError("유효한 경기 행이 없습니다.")
    feat = pd.DataFrame(rows)
    feat["clip_capacity"] = [
        venue_clip_capacity(str(r["구장"]), str(r["홈팀"]), cap_map)
        for _, r in sched.iterrows()
    ]
    if "관중수" in sched.columns:
        feat["관중수"] = pd.to_numeric(sched["관중수"], errors="coerce")
    return feat


def schedule_template_path(root: Path) -> Path:
    return root / "data" / "external" / "batch_predict_schedule_template.csv"
