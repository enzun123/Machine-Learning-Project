# streamlit_app.py

import hashlib
import html
import logging
import re
import os
import sys

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)

_RE_KMA_QSECRET = re.compile(r"((?:authKey|serviceKey)=)([^&\s#'\"]+)", re.I)


def _redact_kma_secret_str(text: object) -> str:
    """HTTP 예외·URL에 붙은 API 인증 쿼리 마스킹 (kv 모듈과 독립)."""
    if text is None:
        return ""
    return _RE_KMA_QSECRET.sub(r"\1***", str(text))


# =========================
# 한글 폰트 (OS별 후보)
# =========================
_font_names = {f.name for f in fm.fontManager.ttflist}
for _fam in (
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Malgun Gothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "DejaVu Sans",
):
    if _fam in _font_names:
        plt.rcParams["font.family"] = _fam
        break
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="KBO 관람 수요 예측 시스템",
    layout="wide"
)

# =========================
# 데이터 불러오기
# =========================
@st.cache_data
def load_data():

    BASE_DIR = Path(__file__).resolve().parent
    root = BASE_DIR.parent.parent

    possible_paths = [

        # 현재 폴더
        BASE_DIR / "kbo_2025_attendance_weather.csv",

        # 상위 폴더들
        BASE_DIR.parent / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent.parent / "kbo_2025_attendance_weather.csv",

        # data 폴더
        root / "data" / "kbo_2025_attendance_weather.csv",
        root / "data" / "interim" / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent.parent / "data" / "kbo_2025_attendance_weather.csv",
    ]

    DATA_PATH = None

    for path in possible_paths:

        if path.exists():
            DATA_PATH = path
            break

    if DATA_PATH is None:

        st.error(
            "CSV 파일을 찾을 수 없습니다.\n"
            "kbo_2025_attendance_weather.csv 파일을 "
            "프로젝트 폴더 또는 data 폴더에 넣어주세요."
        )

        st.stop()

    df = pd.read_csv(DATA_PATH)

    _need = {"경기날짜", "홈팀", "방문팀", "구장", "관중수"}
    if not _need.issubset(df.columns):
        missing = sorted(_need - set(df.columns))
        st.error(
            "CSV에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )
        st.stop()

    df["경기날짜"] = pd.to_datetime(
        df["경기날짜"],
        errors="coerce"
    )

    df["일합계강수량(mm)"] = (
        df["일합계강수량(mm)"]
        .fillna(0)
    )

    return df


def _default_home_team_for_stadium(stadium_name: str, attendance_df: pd.DataFrame) -> str:
    """구장 변경 시 기본 홈팀. 잠실은 LG·두산 공유 → LG 우선."""
    st_key = str(stadium_name).strip()
    homes = sorted(attendance_df["홈팀"].dropna().unique())
    explicit = {
        "잠실": "LG",
        "고척": "키움",
        "수원": "KT",
        "문학": "SSG",
        "인천": "SSG",
        "광주": "KIA",
        "대구": "삼성",
        "창원": "NC",
        "한밭": "한화",
        "대전": "한화",
        "사직": "롯데",
        "부산": "롯데",
        "청주": "한화",
        "포항": "삼성",
        "울산": "NC",
    }
    if st_key in explicit:
        t = explicit[st_key]
        if t in homes:
            return t
    sub = attendance_df.loc[attendance_df["구장"] == st_key, "홈팀"].dropna()
    if len(sub) > 0:
        m = sub.mode()
        if len(m) > 0:
            v = str(m.iloc[0])
            if v in homes:
                return v
    return homes[0] if homes else ""


df = load_data()

_SCRIPTS = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _fallback_stadium_capacity() -> dict[str, int]:
    """CSV를 못 읽을 때만 사용 (kbo_stadium_info.csv와 동기 유지 권장)."""
    return {
        "잠실": 23750,
        "고척": 22258,
        "인천": 23000,
        "문학": 23000,
        "사직": 22669,
        "부산": 22669,
        "창원": 22112,
        "울산": 22000,
        "청주": 10500,
        "포항": 9000,
        "광주": 20500,
        "대전": 13000,
        "한밭": 12000,
        "대구": 29178,
        "수원": 18700,
    }


@st.cache_data
def load_stadium_capacity_map() -> dict[str, int]:
    BASE_DIR = Path(__file__).resolve().parent
    root = BASE_DIR.parent.parent
    for path in (
        root / "data" / "external" / "kbo_stadium_info.csv",
        BASE_DIR.parent.parent.parent / "data" / "external" / "kbo_stadium_info.csv",
    ):
        if not path.exists():
            continue
        st_df = pd.read_csv(path)
        if not {"구장", "최대수용인원"}.issubset(st_df.columns):
            continue
        from common.stadium_aliases import STADIUM_ALIAS

        s = st_df.copy()
        s["구장"] = s["구장"].astype(str).replace(STADIUM_ALIAS)
        s["최대수용인원"] = pd.to_numeric(s["최대수용인원"], errors="coerce")
        g = s.dropna(subset=["구장", "최대수용인원"]).groupby("구장")["최대수용인원"].max()
        out = g.astype(int).to_dict()
        if out:
            return out
    return _fallback_stadium_capacity()


try:
    _KBO_RECENT_TTL = int(os.environ.get("KBO_APP_RECENT_TTL_SEC", "900"))
except ValueError:
    _KBO_RECENT_TTL = 900


@st.cache_data(
    ttl=_KBO_RECENT_TTL,
    show_spinner="KBO 기록실에서 최근 경기를 불러오는 중…",
)
def _recent_five_kbo(stadium: str, before_iso: str) -> pd.DataFrame:
    try:
        from data_collection.fetch_recent_crowd import fetch_recent_games

        headless = os.environ.get("KBO_SCRAPE_HEADLESS", "1") != "0"
        out = fetch_recent_games(
            stadium,
            n=5,
            before=before_iso,
            headless=headless,
        )
        return out if out is not None else pd.DataFrame()
    except Exception:
        logger.warning(
            "KBO 최근 경기 스크랩 실패 (stadium=%s, before=%s)",
            stadium,
            before_iso,
            exc_info=True,
        )
        return pd.DataFrame()


CAP_BY_STADIUM = load_stadium_capacity_map()


def get_capacity(stadium_name: str) -> int:
    from common.stadium_aliases import STADIUM_ALIAS

    raw = str(stadium_name).strip()
    if not raw:
        return 20000
    norm = STADIUM_ALIAS.get(raw, raw)
    if norm in CAP_BY_STADIUM:
        return int(CAP_BY_STADIUM[norm])
    if raw in CAP_BY_STADIUM:
        return int(CAP_BY_STADIUM[raw])
    for gu, c in CAP_BY_STADIUM.items():
        if gu in raw or raw in gu:
            return int(c)
    return 20000


# ----- ML (attendance_rf_pipeline.joblib) -----
RAIN_BINS_ML = [-1.0, 0.0, 1.0, 5.0, float("inf")]
RAIN_LABELS_ML = ["No_Rain", "Rain_0_1mm", "Rain_1_5mm", "Rain_5mm_plus"]
_SMALL_STADIUM_ML = 15_000
_DERBY_PAIRS_ML = {
    frozenset({"LG", "두산"}),
    frozenset({"롯데", "NC"}),
    frozenset({"삼성", "KIA"}),
    frozenset({"KT", "키움"}),
    frozenset({"SSG", "한화"}),
}


def _scalar_rain_bucket_ml(mm: float) -> str:
    s = pd.cut(
        pd.Series([float(mm)]),
        bins=RAIN_BINS_ML,
        labels=RAIN_LABELS_ML,
        include_lowest=True,
    )
    v = s.iloc[0]
    if pd.isna(v):
        return "No_Rain"
    return str(v)


def _scalar_temp_bucket_ml(t: float) -> str:
    s = pd.cut(
        pd.Series([float(t)]),
        bins=[-float("inf"), 10, 20, 25, 30, float("inf")],
        labels=["VeryCold", "Cold", "Mild", "Warm", "Hot"],
    )
    v = s.iloc[0]
    if pd.isna(v):
        return "Mild"
    return str(v)


def _scalar_hum_bucket_ml(h: float) -> str:
    s = pd.cut(
        pd.Series([float(h)]),
        bins=[0, 40, 60, 80, 100],
        labels=["Dry", "Normal", "Humid", "VeryHumid"],
    )
    v = s.iloc[0]
    if pd.isna(v):
        return "Normal"
    return str(v)


def _pick_template_series_ml(tr: pd.DataFrame, home: str, away: str, stadium: str) -> pd.Series:
    from common.stadium_aliases import STADIUM_ALIAS

    g = tr.copy()
    g["_g"] = g["구장"].astype(str).replace(STADIUM_ALIAS)
    st_n = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    hs, vs = str(home), str(away)
    masks = [
        (g["홈팀"].astype(str) == hs) & (g["방문팀"].astype(str) == vs) & (g["_g"] == st_n),
        (g["홈팀"].astype(str) == hs) & (g["_g"] == st_n),
        g["_g"] == st_n,
    ]
    for m in masks:
        sub = g.loc[m]
        if len(sub) >= 1:
            return sub.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]
    return g.sort_values(["연도", "월", "주차_ISO"]).iloc[-1]


def _build_ml_feature_row(
    tr: pd.DataFrame,
    home: str,
    away: str,
    stadium: str,
    game_date,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
    cap: int,
) -> pd.DataFrame:
    from common.stadium_aliases import STADIUM_ALIAS
    from modeling.train_model import FEATURE_COLUMNS

    row = _pick_template_series_ml(tr, home, away, stadium)
    gdt = pd.Timestamp(game_date)
    wdn = int(gdt.dayofweek)
    st_key = STADIUM_ALIAS.get(str(stadium).strip(), str(stadium).strip())
    is_rain_i = int(rain_mm > 0)
    rain_b = _scalar_rain_bucket_ml(rain_mm)
    temp_b = _scalar_temp_bucket_ml(temp_c)
    hum_b = _scalar_hum_bucket_ml(hum_pct)
    derby = int(frozenset({str(home), str(away)}) in _DERBY_PAIRS_ML)

    d = {c: row.get(c, np.nan) for c in FEATURE_COLUMNS}
    upd = {
        "연도": int(gdt.year),
        "월": int(gdt.month),
        "주차_ISO": int(gdt.isocalendar().week),
        "홈팀": str(home),
        "방문팀": str(away),
        "구장": st_key,
        "stadium_capacity": float(cap),
        "is_capacity_missing": 0,
        "is_rain": is_rain_i,
        "rain_bucket": rain_b,
        "temp_bucket": temp_b,
        "is_hot": int(temp_c >= 30),
        "humidity_bucket": hum_b,
        "is_weekend": int(wdn >= 5),
        "is_friday": int(wdn == 4),
        "is_saturday": int(wdn == 5),
        "is_sunday": int(wdn == 6),
        "weekday_sin": float(np.sin(2 * np.pi * wdn / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * wdn / 7)),
        "stadium_x_rain": f"{st_key}_{is_rain_i}",
        "is_small_stadium": int(cap < _SMALL_STADIUM_ML),
        "is_derby": derby,
        "is_childrens_day": int(gdt.month == 5 and gdt.day in (4, 5, 6)),
    }
    for k, v in upd.items():
        if k in d:
            d[k] = v
    return pd.DataFrame([d])


def _aggregate_rf_importance_from_pipe(pipe) -> pd.Series:
    """OneHot+수치 파이프라인에서 원본 피처명 기준으로 중요도 합산."""
    from modeling.train_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    reg = pipe.regressor_
    pre = reg.named_steps["preprocess"]
    rf = reg.named_steps["model"]
    s = pd.Series(rf.feature_importances_, index=pre.get_feature_names_out())
    out: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        k = f"num__{col}"
        if k in s.index:
            out[col] = float(s[k])
    for col in CATEGORICAL_FEATURES:
        p = f"cat__{col}_"
        m = s.index.astype(str).str.startswith(p)
        out[col] = float(s.loc[m].sum()) if m.any() else 0.0
    return pd.Series(out).sort_values(ascending=False)


@st.cache_data(show_spinner=False)
def _cached_rf_feature_importance_series(model_path_str: str, mtime_key: int) -> pd.Series:
    p = Path(model_path_str)
    if not p.exists():
        return pd.Series(dtype=float)
    pipe = joblib.load(p)
    return _aggregate_rf_importance_from_pipe(pipe)


_ML_FEAT_LABEL_KO: dict[str, str] = {
    "연도": "연도",
    "월": "월",
    "주차_ISO": "주차(ISO)",
    "stadium_capacity": "구장 정원",
    "is_capacity_missing": "정원 결측 여부",
    "is_rain": "강수(0/1)",
    "is_hot": "폭염(30℃+)",
    "is_weekend": "주말 여부",
    "is_friday": "금요일",
    "is_saturday": "토요일",
    "is_sunday": "일요일",
    "weekday_sin": "요일 sin",
    "weekday_cos": "요일 cos",
    "is_small_stadium": "소형 구장 여부",
    "is_derby": "더비 매치",
    "is_season_opener": "시즌 초반(오프너)",
    "is_childrens_day": "어린이날·인접일",
    "home_win_rate": "홈팀 승률(시점)",
    "visitor_win_rate": "원정 승률(시점)",
    "win_rate_diff": "승률 차",
    "home_gb_to_5th": "홈 5위와 게임차",
    "visitor_gb_to_5th": "원정 5위와 게임차",
    "is_pennant_race": "페넌트레이스 구간",
    "playoff_urgency": "플레이오프 긴박도",
    "month_x_playoff_urgency": "월 × 긴박도",
    "home_prior_mean_att": "홈팀 과거 평균 관중",
    "visitor_prior_mean_att": "원정 과거 평균 관중",
    "home_last5_mean_att": "홈 최근 5경기 평균 관중",
    "visitor_last5_mean_att": "원정 최근 5경기 평균 관중",
    "home_draw_pct_in_league": "홈 리그 무승부 비율",
    "visitor_away_draw_pct_in_league": "원정 원정 무승부 비율",
    "home_visitor_prior_draw_diff": "무승부 비율 차",
    "matchup_prior_mean_att": "매치업 과거 평균 관중",
    "season_progress": "시즌 진행도",
    "홈팀": "홈팀(범주)",
    "방문팀": "원정팀(범주)",
    "구장": "구장(범주)",
    "rain_bucket": "강수 구간(범주)",
    "temp_bucket": "기온 구간(범주)",
    "humidity_bucket": "습도 구간(범주)",
    "wind_bucket": "풍속 구간(범주)",
    "stadium_x_rain": "구장×강수(범주)",
    "날씨(강수·구장연동)": "날씨(강수·구장연동)",
    "날씨(기온·습도·풍)": "날씨(기온·습도·풍)",
}

_ML_IMP_RAIN_GROUP = frozenset({"rain_bucket", "is_rain", "stadium_x_rain"})
_ML_IMP_THERMO_GROUP = frozenset({"temp_bucket", "humidity_bucket", "wind_bucket", "is_hot"})
_ML_IMP_WEATHER_DISPLAY_KEYS = ("날씨(강수·구장연동)", "날씨(기온·습도·풍)")

_TEMP_BUCKET_KO: dict[str, str] = {
    "VeryCold": "매우 쌀쌀",
    "Cold": "쌀쌀",
    "Mild": "쾌적",
    "Warm": "따뜻",
    "Hot": "더움",
}
_RAIN_BUCKET_KO: dict[str, str] = {
    "No_Rain": "무강수",
    "Rain_0_1mm": "미세 강수(~1mm)",
    "Rain_1_5mm": "소강수(1~5mm)",
    "Rain_5mm_plus": "강한 강수(5mm+)",
}
_HUM_BUCKET_KO: dict[str, str] = {
    "Dry": "건조",
    "Normal": "보통",
    "Humid": "다습",
    "VeryHumid": "매우 다습",
}
_WIND_BUCKET_KO: dict[str, str] = {
    "Calm": "잔풍",
    "Light": "약풍",
    "Moderate": "보통",
    "Strong": "강풍",
}


def _ko_ml_feature_label(name: str) -> str:
    return _ML_FEAT_LABEL_KO.get(str(name), str(name))


def _ml_cell_display(v: object) -> object:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
            return round(x, 4)
        return round(x, 6)
    return str(v)


def _group_rf_importance_for_display(imp: pd.Series) -> pd.Series:
    """날씨 세부 피처를 두 그룹으로 합산해 막대·표에서 한눈에 보이게 함."""
    s = imp.astype(float)
    rain_keys = [k for k in _ML_IMP_RAIN_GROUP if k in s.index]
    therm_keys = [k for k in _ML_IMP_THERMO_GROUP if k in s.index]
    g_r = float(s[rain_keys].sum()) if rain_keys else 0.0
    g_t = float(s[therm_keys].sum()) if therm_keys else 0.0
    rest = s.drop(labels=rain_keys + therm_keys, errors="ignore")
    extra = pd.Series(
        {"날씨(강수·구장연동)": g_r, "날씨(기온·습도·풍)": g_t},
        dtype=float,
    )
    out = pd.concat([rest, extra])
    return out.sort_values(ascending=False)


def _ml_prediction_snapshot(
    row: pd.Series,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for c in (
        "홈팀",
        "방문팀",
        "구장",
        "연도",
        "월",
        "주차_ISO",
        "is_weekend",
        "stadium_capacity",
        "is_derby",
        "matchup_prior_mean_att",
        "home_prior_mean_att",
        "visitor_prior_mean_att",
        "home_win_rate",
        "visitor_win_rate",
    ):
        if c not in row.index:
            continue
        out[_ko_ml_feature_label(c)] = _ml_cell_display(row[c])

    rb = str(row.get("rain_bucket", "") or "").strip()
    tb = str(row.get("temp_bucket", "") or "").strip()
    hb = str(row.get("humidity_bucket", "") or "").strip()
    wb_raw = row.get("wind_bucket")
    if wb_raw is None or pd.isna(wb_raw):
        wb = ""
    else:
        wb = str(wb_raw).strip()
    out["강수(구간·입력)"] = f"{_RAIN_BUCKET_KO.get(rb, rb)} ({rain_mm:g}mm)"
    out["기온(구간·입력)"] = f"{_TEMP_BUCKET_KO.get(tb, tb)} ({temp_c:g}℃)"
    out["습도(구간·입력)"] = f"{_HUM_BUCKET_KO.get(hb, hb)} ({hum_pct:g}%)"
    out["풍속(구간)"] = _WIND_BUCKET_KO.get(wb, wb or "—")
    return out


def _plot_rf_importance_barh(imp: pd.Series, top_n: int = 15) -> plt.Figure:
    """비날씨는 상위 top_n만, 날씨 두 그룹은 값이 작아도 항상 막대에 포함."""
    s = imp.astype(float)
    rest = s.drop(labels=list(_ML_IMP_WEATHER_DISPLAY_KEYS), errors="ignore").sort_values(
        ascending=False
    )
    others_top = rest.head(top_n)
    weather = pd.Series(
        {k: float(s[k]) if k in s.index else 0.0 for k in _ML_IMP_WEATHER_DISPLAY_KEYS},
        dtype=float,
    )
    others_asc = others_top.sort_values(ascending=True)
    weather_asc = weather.sort_values(ascending=True)
    tail = pd.concat([others_asc, weather_asc])
    labels = [_ko_ml_feature_label(str(i)) for i in tail.index]
    vals = tail.to_numpy(dtype=float)
    tot = float(vals.sum()) or 1.0
    pct = vals / tot * 100.0

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(tail))))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")
    y = np.arange(len(tail))
    ax.barh(y, pct, color="#4f8cff", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#e8eef5", fontsize=10)
    ax.set_xlabel("상대 기여 (전체 중 %)", color="#9fb3c8", fontsize=11)
    ax.tick_params(axis="x", colors="#9fb3c8")
    ax.set_title(
        "RandomForest 피처 중요도 (날씨 세부는 2그룹으로 합산)",
        color="white",
        fontsize=13,
    )
    for spine in ax.spines.values():
        spine.set_color("#24384f")
    fig.tight_layout()
    return fig


@st.cache_resource
def _load_rf_pipeline():
    p = PROJECT_ROOT / "models" / "attendance_rf_pipeline.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_data
def _load_kbo_train_ready() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(ttl=600)
def _cached_forecast_rain_ref(stadium: str, game_start_iso: str, _auth_fp: str):
    """동네예보(typ02) 개시 3시간 전 참고. 키는 캐시용 지문만 사용."""
    from common import kma_vilage_fcst as kv

    auth = os.environ.get("KMA_APIHUB_AUTH_KEY", "").strip()
    if not auth:
        try:
            auth = str(st.secrets.get("KMA_APIHUB_AUTH_KEY", "")).strip()
        except Exception:
            pass
    gs = pd.Timestamp(game_start_iso)
    if gs.tzinfo is None:
        gs = gs.tz_localize(kv.KST)
    else:
        gs = gs.tz_convert(kv.KST)
    return kv.forecast_ref_for_rain_cancel_rules(stadium, gs, auth_key=auth or None)


# =========================
# CSS 스타일
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #07111f;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #0b1726;
    border-right: 1px solid #24384f;
}

.main-title {
    font-size: 32px;
    font-weight: 900;
    color: white;
    margin-bottom: 8px;
}

.sub-text {
    color: #9fb3c8;
    font-size: 15px;
}

.match-text {
    color: white;
    font-size: 18px;
    margin-top: 18px;
    margin-bottom: 18px;
}

.predict-box {
    border: 1px dashed #18e6ff;
    border-radius: 14px;
    padding: 45px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 25px;
    background-color: #081525;
}

.predict-label {
    color: white;
    font-size: 18px;
    margin-bottom: 10px;
}

.predict-number {
    color: #18e6ff;
    font-size: 62px;
    font-weight: 900;
}

.card {
    background-color: #101d2d;
    border: 1px solid #263f5c;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.card-title {
    color: #b8c7d9;
    font-size: 17px;
    margin-bottom: 10px;
}

.card-value {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.card-high {
    color: #ff4b5c;
    font-size: 38px;
    font-weight: 900;
}

.action-low {
    background-color: #0f3425;
    border: 1px solid #2ecc71;
    border-radius: 15px;
    padding: 25px;
    color: white;
}

.action-mid {
    background-color: #3c3414;
    border: 1px solid #f1c40f;
    border-radius: 15px;
    padding: 25px;
    color: white;
}

.action-high {
    background-color: #3b1116;
    border: 1px solid #ff4b5c;
    border-radius: 15px;
    padding: 25px;
    color: white;
}

.disclaimer-under-predict {
    background-color: #101d2d;
    border: 1px solid #2a4a6f;
    border-radius: 12px;
    padding: 16px 18px;
    margin: 16px 0 20px 0;
    font-size: 13px;
    line-height: 1.65;
    color: #c5d8ec;
}
.disclaimer-under-predict b {
    color: #e8f0f8;
}
.disclaimer-under-predict code {
    font-size: 12px;
    color: #b8d4f0;
}

.rain-risk-box {
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 14px;
    margin-bottom: 8px;
    font-size: 14px;
    line-height: 1.55;
    color: #e8eef5;
}

.rain-risk-low {
    border: 1px solid #2ecc71;
    background: rgba(46, 204, 113, 0.09);
}

.rain-risk-mid {
    border: 1px solid #f39c12;
    background: rgba(243, 156, 18, 0.12);
}

.rain-risk-high {
    border: 1px solid #e74c3c;
    background: rgba(231, 76, 60, 0.11);
}

.rain-fcst-warn {
    border: 1px solid #c9a227;
    background: rgba(201, 162, 39, 0.14);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 14px;
    margin-bottom: 8px;
    font-size: 14px;
    line-height: 1.55;
    color: #e8eef5;
}
.rain-fcst-warn-title {
    font-weight: 700;
    margin-bottom: 8px;
    color: #f5e6a8;
}
.rain-fcst-warn-tech {
    font-size: 12px;
    color: #9fb3c8;
    margin: 0;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 사이드바
# =========================
st.sidebar.title("⚾ KBO Attendance Predictor")

game_date = st.sidebar.date_input("경기 날짜")
st.sidebar.caption(
    "차트·최근 경기 자동 반영·예측 입력의 **기준일**입니다. 당일 0시 **이전** 경기만 포함합니다."
)

_stadium_opts = sorted(df["구장"].dropna().unique())
_home_opts = sorted(df["홈팀"].dropna().unique())
_away_opts = sorted(df["방문팀"].dropna().unique())

stadium = st.sidebar.selectbox(
    "경기장",
    _stadium_opts,
    key="fld_stadium",
    help="구장을 바꾸면 **홈팀**이 이 구장의 기본 홈(잠실→LG 등)으로 맞춰집니다.",
)

if "_prev_stadium_for_home" not in st.session_state:
    st.session_state._prev_stadium_for_home = None
if st.session_state._prev_stadium_for_home != stadium:
    st.session_state._prev_stadium_for_home = stadium
    _dh = _default_home_team_for_stadium(stadium, df)
    if _dh in _home_opts:
        st.session_state.fld_home_team = _dh
    _cur_away = st.session_state.get("fld_away_team")
    if _cur_away is None or _cur_away == st.session_state.get("fld_home_team"):
        for _a in _away_opts:
            if _a != st.session_state.get("fld_home_team"):
                st.session_state.fld_away_team = _a
                break

home_team = st.sidebar.selectbox(
    "홈팀",
    _home_opts,
    key="fld_home_team",
)

away_team = st.sidebar.selectbox(
    "원정팀",
    _away_opts,
    key="fld_away_team",
)

auto_recent_kbo = st.sidebar.checkbox(
    "최근 5경기 KBO 자동 반영",
    value=os.environ.get("STREAMLIT_WEB_RECENT", "1") != "0",
    help=(
        "선택한 경기 날짜 이전에 치른 직전 5경기를 GraphDaily에서 가져옵니다. "
        "Chrome·Selenium 필요. 꺼두면 로컬 CSV만 사용합니다. "
        f"캐시 TTL {_KBO_RECENT_TTL}초."
    ),
)

st.sidebar.markdown("---")

_rf_model_path = PROJECT_ROOT / "models" / "attendance_rf_pipeline.joblib"
_rf_train_path = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
_rf_artifacts_ok = _rf_model_path.exists() and _rf_train_path.exists()

temperature = st.sidebar.slider(
    "예상 기온(℃)",
    -10,
    40,
    23
)

rainfall_mm = st.sidebar.slider(
    "일 합계 강수(mm)",
    0.0,
    120.0,
    0.0,
    0.5,
    help=(
        "**RandomForest** 입력의 `rain_bucket`·`is_rain`·`stadium_x_rain` 등에 반영됩니다. "
        "휴리스틱만 사용할 때는 이 슬라이더가 관중 추정에 직접 쓰이지 않을 수 있습니다."
    ),
)

humidity = st.sidebar.slider(
    "예상 습도(%)",
    0,
    100,
    60
)

use_rf_model = st.sidebar.checkbox(
    "RandomForest 관중 예측",
    value=_rf_artifacts_ok,
    disabled=not _rf_artifacts_ok,
    help=(
        "켜면: 사이드바의 날짜·기온·습도·**일 강수(mm)**·구장 정원 등을 반영해 학습된 RandomForest로 관중을 추정합니다. "
        "끄면: 과거 평균과 간단 룰(폭염·고습도 등)로 추정합니다.\n\n"
        "개발·운영 참고: 입력 행은 `data/processed/kbo_train_ready.csv`에서 홈·원정·구장 유사도로 고릅니다. "
        "강수량은 구간(`rain_bucket` 등)으로 변환되어 모델에 들어갑니다. "
        "필요 파일: `models/attendance_rf_pipeline.joblib`, 위 CSV. 없거나 오류 시 휴리스틱만 사용합니다."
    ),
)

# =========================
# 예측: 휴리스틱 + (옵션) RF 파이프라인
# =========================
home_slice = df[
    (df["구장"] == stadium)
    & (df["홈팀"] == home_team)
]
visitor_slice = df[
    (df["구장"] == stadium)
    & (df["방문팀"] == away_team)
]
matchup_slice = df[
    (df["구장"] == stadium)
    & (df["홈팀"] == home_team)
    & (df["방문팀"] == away_team)
]

pred_source = ""

if len(matchup_slice) >= 1:

    pred_source = "matchup"
    predicted_heuristic = int(
        matchup_slice["관중수"].mean()
    )

elif len(home_slice) >= 1 and len(visitor_slice) >= 1:

    pred_source = "blend"
    predicted_heuristic = int(
        (
            home_slice["관중수"].mean()
            + visitor_slice["관중수"].mean()
        )
        / 2
    )

elif len(home_slice) >= 1:

    pred_source = "home_only"
    predicted_heuristic = int(
        home_slice["관중수"].mean()
    )

elif len(visitor_slice) >= 1:

    pred_source = "visitor_only"
    predicted_heuristic = int(
        visitor_slice["관중수"].mean()
    )

else:

    stadium_only = df[df["구장"] == stadium]

    if len(stadium_only) > 0:

        pred_source = "stadium_only"
        predicted_heuristic = int(
            stadium_only["관중수"].mean()
        )

    else:

        pred_source = "global"
        predicted_heuristic = int(
            df["관중수"].mean()
        )

# 휴리스틱: 강수 시 관중 약간 하향(RF의 rain_bucket과 별개; 일 강수 슬라이더 > 0일 때만).
if rainfall_mm > 0:
    predicted_heuristic = int(predicted_heuristic * 0.93)

if humidity >= 85:

    predicted_heuristic = int(
        predicted_heuristic * 0.98
    )

stadium_capacity = get_capacity(stadium)

predicted_heuristic = min(
    predicted_heuristic,
    max(1, int(stadium_capacity * 1.05)),
)

ml_used = False
ml_row_snapshot: dict[str, object] | None = None
predicted_attendance = predicted_heuristic

if use_rf_model and _rf_artifacts_ok:

    pipe = _load_rf_pipeline()
    tr = _load_kbo_train_ready()

    if pipe is not None and tr is not None:

        try:

            from modeling.train_model import FEATURE_COLUMNS

            X = _build_ml_feature_row(
                tr,
                home_team,
                away_team,
                stadium,
                game_date,
                float(temperature),
                float(rainfall_mm),
                float(humidity),
                stadium_capacity,
            )
            X = X[FEATURE_COLUMNS].copy()
            for col in X.columns:

                if X[col].dtype == object:

                    X[col] = X[col].astype(str).fillna("missing")

            raw_ml = float(np.asarray(pipe.predict(X))[0])
            predicted_attendance = int(round(max(0.0, raw_ml)))
            predicted_attendance = min(
                predicted_attendance,
                max(1, int(stadium_capacity * 1.05)),
            )
            ml_used = True
            ml_row_snapshot = _ml_prediction_snapshot(
                X.iloc[0],
                float(temperature),
                float(rainfall_mm),
                float(humidity),
            )

        except Exception:

            logger.warning("RF 파이프라인 예측 실패, 휴리스틱 유지", exc_info=True)

if ml_used:

    st.caption("예측에 **RandomForest** 모델을 반영했습니다.")

    with st.expander("피처 중요도 · 이번 입력 요약 (RandomForest)", expanded=False):
        st.markdown(
            "아래 **막대 그래프**는 학습된 숲 전체에서 평균적으로 분할에 자주 쓰인 변수입니다. "
            "강수·기온·습도·풍 세부 피처는 중요도 표에서 **두 줄(날씨 그룹)** 로 합산했습니다. "
            f"지금 화면의 **{predicted_attendance:,}명** 같은 **한 건의 예측**을 인과적으로 쪼개는 값(SHAP 등)은 아니며, "
            "모델이 전반적으로 어떤 정보에 무게를 두었는지 참고용입니다."
        )
        try:
            _mt = int(os.path.getmtime(_rf_model_path))
        except OSError:
            _mt = 0
        _imp = _cached_rf_feature_importance_series(str(_rf_model_path), _mt)
        if len(_imp) > 0:
            _imp_disp = _group_rf_importance_for_display(_imp)
            _fig_imp = _plot_rf_importance_barh(_imp_disp, top_n=15)
            st.pyplot(_fig_imp)
            plt.close(_fig_imp)
            _p_all = (_imp_disp / _imp_disp.sum() * 100.0).round(2)
            _w_fix = _p_all.reindex(list(_ML_IMP_WEATHER_DISPLAY_KEYS)).fillna(0.0)
            _rest_tbl = (
                _p_all.drop(labels=list(_ML_IMP_WEATHER_DISPLAY_KEYS), errors="ignore")
                .sort_values(ascending=False)
                .head(20)
            )
            _pct = pd.concat([_w_fix, _rest_tbl])
            _tbl = _pct.reset_index()
            _tbl.columns = ["피처", "기여(%)"]
            _tbl["피처"] = _tbl["피처"].map(lambda x: _ko_ml_feature_label(str(x)))
            st.dataframe(_tbl, width="stretch", hide_index=True)
        else:
            st.info("피처 중요도를 불러오지 못했습니다.")

        if ml_row_snapshot:
            st.markdown("**이번 예측에 넣은 주요 값** (유사 과거 행 + 사이드바 일부 덮어쓴 뒤)")
            _snap_rows = []
            for k, v in ml_row_snapshot.items():
                if isinstance(v, bool):
                    _disp = "예" if v else "아니오"
                else:
                    _disp = str(v)
                _snap_rows.append({"항목": k, "값": _disp})
            st.dataframe(
                pd.DataFrame(_snap_rows),
                width="stretch",
                hide_index=True,
            )

else:

    if pred_source in ("stadium_only", "global"):

        st.info(
            "선택한 팀·구장 조합의 과거 경기가 적어 "
            "**구장 또는 리그 전체 평균**으로 추정했습니다."
        )

    elif pred_source == "visitor_only":

        st.caption(
            "해당 구장에서 홈으로 치른 **홈팀** 기록이 없어, "
            "**원정팀 방문** 기록 위주로 추정했습니다."
        )

    elif pred_source == "home_only":

        st.caption(
            "해당 구장에서 **원정팀**이 온 기록이 없어, "
            "**홈팀 홈** 기록만으로 추정했습니다."
        )

    elif pred_source == "blend":

        st.caption(
            "동일 매치업 기록이 없어, **홈팀 홈 평균**과 "
            "**원정팀 방문 평균**을 반반 반영했습니다."
        )

    elif pred_source == "matchup":

        st.caption(
            "동일 **홈·원정·구장** 조합의 과거 관중 평균을 사용했습니다."
        )

    if use_rf_model and not _rf_artifacts_ok:

        st.caption(
            "RandomForest를 켰지만 필요한 모델·데이터 파일이 없어 **휴리스틱만** 사용했습니다. "
            "체크박스 도움말(?)을 확인하세요."
        )

# =========================
# 혼잡도 계산
# =========================
congestion = (
    predicted_attendance /
    stadium_capacity
) * 100

# =========================
# 운영 단계
# =========================
if congestion < 50:

    level = "LOW"

    action_class = "action-low"

    action_title = "🟢 [비용 절감 모드]"

    action_msg = (
        "식자재 발주를 평소 대비 축소하고, "
        "일부 구역 매점 운영을 최소화하여 "
        "인력을 효율적으로 운용하세요."
    )

elif congestion < 80:

    level = "NORMAL"

    action_class = "action-mid"

    action_title = "🟡 [일반 운영 모드]"

    action_msg = (
        "기본 매뉴얼에 따라 운영을 준비하세요. "
        "입장 동선과 매점 운영 상태를 "
        "사전에 점검하세요."
    )

else:

    level = "HIGH"

    action_class = "action-high"

    action_title = "🔴 [안전 강화 모드]"

    action_msg = (
        "게이트 주변 안전 요원을 "
        "20% 추가 배치하고, "
        "매점 재고 소진에 대비해 "
        "발주량을 최대로 늘리세요."
    )

# =========================
# 메인 화면
# =========================
st.markdown(
    '<div class="main-title">📈 KBO 관람 수요 예측 시스템</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">'
    "사이드바에서 켜면 <b>RandomForest 파이프라인</b> "
    "<code>models/attendance_rf_pipeline.joblib</code>으로 예측하고, "
    "끄면 <b>과거 CSV 평균 + 날씨 룰</b>만 사용합니다. "
    "최근 5경기 차트는 옵션에 따라 KBO 기록실에서 갱신할 수 있습니다."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown("## 📅 경기 정보")

st.markdown(
    f'<div class="match-text">'
    f'{game_date} | '
    f'{home_team} vs {away_team} | '
    f'{stadium}'
    f'</div>',
    unsafe_allow_html=True
)

# =========================
# 예상 관중 수
# =========================
st.markdown(f"""
<div class="predict-box">

<div class="predict-label">
예상 관중 수
</div>

<div class="predict-number">
{predicted_attendance:,} 명
</div>

</div>
""", unsafe_allow_html=True)

_wx_debug_ui = os.environ.get("STREAMLIT_DEBUG_WEATHER", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

from common.kbo_regular_start_time import default_start_hm
from common import kma_vilage_fcst as kv

st.markdown(
    """
<div class="disclaimer-under-predict">
<b>면책·안내.</b>
예상 관중은 <b>경기가 정상 개최된다는 전제</b>의 수요 추정이며,
우천 취소·노게임·강우 콜드게임 등에는 적용되지 않습니다.
동네예보(typ02)는 KBO·보도 문구와 <b>대조용 참고</b>일 뿐, 관중 예측 수치에는 넣지 않습니다.
RandomForest 사용 시 사이드바 <b>일 합계 강수(mm)</b>·기온·습도는 모델 입력(버킷 등)에 반영됩니다.
구장 정원은 <code>kbo_stadium_info.csv</code>를 우선합니다.
위 요약은 안내용이며 실제 운영·규정과 다를 수 있습니다.
</div>
""",
    unsafe_allow_html=True,
)

_kbo_col, _typ_col = st.columns((1, 1))
with _kbo_col:
    st.markdown(
        """
##### KBO에서 흔히 인용되는 검토 시점

- **개시 3시간 전**: 시간당 **10mm 이상 예보** 등을 두고 취소·연기를 검토한다는 설명이 많습니다.  
- **개시 1시간 전**: 시간당 **5mm 이상 실제 강우** 검토가 거론됩니다.  
- **경기 중** 강우 중단·콜드게임·노게임(5회 전후 등)은 **공식 규정·심판 판단**입니다.

위는 언론·운영 안내 수준이며 실제와 다를 수 있습니다.
        """
    )

with _typ_col:
    st.markdown(
        """
##### 동네예보(typ02) — 개시 3시간 전 참고

KBO에서 말하는 **‘개시 3시간 전’** 은 보통 **예보** 기준입니다. 기상청 API허브 **동네예보(typ02)** 로
**개시 시각 기준 3시간 전**에 가장 가까운 **초단기 1시간 강수(RN1)** 를 불러오고,
안 되면 **단기 강수확률(POP)** 만 표시합니다. API허브에서 **초단기·단기예보** 활용신청이 된 키(`KMA_APIHUB_AUTH_KEY`)가 있어야 합니다.
        """
    )

_h_f, _m_f = default_start_hm(game_date, game_date.year)
_gs_fc = pd.Timestamp(
    year=game_date.year,
    month=game_date.month,
    day=game_date.day,
    hour=_h_f,
    minute=_m_f,
    tz=kv.KST,
)
_t3 = kv.three_hours_before_game_start(_gs_fc)
st.caption(
    f"가정 개시 **{_gs_fc.strftime('%Y-%m-%d %H:%M')}** (KST) · "
    f"참고 시각(개시 3시간 전) **{_t3.strftime('%Y-%m-%d %H:%M')}**"
)
_auth_fc = os.environ.get("KMA_APIHUB_AUTH_KEY", "").strip()
if not _auth_fc:
    try:
        _auth_fc = str(st.secrets.get("KMA_APIHUB_AUTH_KEY", "")).strip()
    except Exception:
        pass
_fp = hashlib.sha256(_auth_fc.encode()).hexdigest()[:16] if _auth_fc else "nokey"
_ref = _cached_forecast_rain_ref(stadium, _gs_fc.isoformat(), _fp)

if _wx_debug_ui:
    logger.info(
        "kma_fcst_ref stadium=%r game_start=%s ref3h=%s payload=%s",
        stadium,
        _gs_fc.isoformat(),
        str(_t3),
        {k: _ref.get(k) for k in ("ok", "mode", "base", "nx", "ny", "detail", "msg", "target_kst")},
    )
    with st.expander("기상·API 디버그 (`STREAMLIT_DEBUG_WEATHER=1`)", expanded=False):
        st.json(_ref)

if not _ref.get("ok"):
    _m = str(_ref.get("msg") or "예보를 가져오지 못했습니다.")
    _d = _ref.get("detail")
    _d_s = str(_d).strip() if _d is not None else ""
    _body = (
        f'<div class="rain-fcst-warn">'
        f'<div class="rain-fcst-warn-title">동네예보 참고</div>'
        f'<p style="margin:0 0 8px 0;">{html.escape(_m)}</p>'
    )
    if _d_s:
        _body += (
            f'<p class="rain-fcst-warn-tech">({html.escape(_redact_kma_secret_str(_d_s))})</p>'
        )
    _body += "</div>"
    st.markdown(_body, unsafe_allow_html=True)
elif _ref.get("mode") == "ultra_rn1":
    mm = float(_ref["mm_h"])
    _t, _b, _css = kv.rule_band_from_mm_h(mm)
    st.markdown(
        f'<div class="rain-risk-box rain-risk-{_css}">'
        f'<div class="rain-risk-title">초단기예보 RN1 ≈ {mm:g} mm/h (1시간 강수)</div>'
        f"<p style=\"margin:0;\"><b>{_t}</b> — {_b}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
elif _ref.get("mode") == "vilage_pop":
    pop = float(_ref["pop_pct"])
    _t, _b, _css = kv.rule_band_from_pop(pop)
    st.markdown(
        f'<div class="rain-risk-box rain-risk-{_css}">'
        f'<div class="rain-risk-title">단기예보 강수확률 POP ≈ {pop:.0f}%</div>'
        f"<p style=\"margin:0;\"><b>{_t}</b> — {_b}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _pop_detail = _ref.get("detail")
    if _pop_detail:
        st.caption(
            "초단기 RN1은 가져오지 못해 단기 POP만 표시합니다 — "
            + html.escape(_redact_kma_secret_str(str(_pop_detail).strip()))
        )

_gout = kv.rainout_cancel_guidance(_ref)
_gband = str(_gout["band"])
_wrap_g = "rain-fcst-warn" if _gband == "warn" else f"rain-risk-box rain-risk-{_gband}"
_gparts: list[str] = [
    f'<div class="{_wrap_g}">',
    f'<div class="rain-risk-title">{html.escape(str(_gout["headline"]))}</div>',
]
_src_g = str(_gout.get("source") or "").strip()
if _src_g:
    _gparts.append(
        '<p style="margin:0 0 10px 0;font-size:12px;color:#9fb3c8;">'
        f"{html.escape(_src_g)}</p>"
    )
for _ln in _gout["lines"]:
    _gparts.append(
        f'<p style="margin:0 0 8px 0;">{html.escape(_redact_kma_secret_str(str(_ln)))}</p>'
    )
_gparts.append("</div>")
st.markdown(
    '<div class="sub-text" style="margin-top:18px;margin-bottom:6px;">'
    "<b>☂️ 우천 취소·연기 참고</b> (개시 3시간 전 동네예보 기준 · 관중 수 예측과 별개)"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("\n".join(_gparts), unsafe_allow_html=True)

# =========================
# 정보 카드
# =========================
col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    👥 경기장 수용 인원
    </div>

    <div class="card-value">
    {stadium_capacity:,} 명
    </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    📊 혼잡도
    </div>

    <div class="card-value">
    {congestion:.1f}%
    </div>

    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    🛡 운영 강도
    </div>

    <div class="card-high">
    {level}
    </div>

    </div>
    """, unsafe_allow_html=True)

# =========================
# 액션 플랜
# =========================
st.markdown("## 🛡 상황별 액션 플랜")

st.markdown(f"""
<div class="{action_class}">

<h2>{action_title}</h2>

<p style="font-size:18px; line-height:1.7;">
{action_msg}
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# 최근 경기 그래프
# =========================
st.markdown(
    "## 📊 해당 경기장 최근 5경기 관중 수 추이"
)

_by_st = df[df["구장"] == stadium].dropna(subset=["경기날짜"])
_local = (
    _by_st[_by_st["경기날짜"] < pd.Timestamp(game_date)]
    .sort_values("경기날짜", ascending=False)
    .head(5)
    .sort_values("경기날짜")
)

recent_games = _local
chart_source = "로컬 CSV"

if auto_recent_kbo:
    scraped = _recent_five_kbo(stadium, game_date.isoformat())
    if scraped is not None and len(scraped) >= 1:
        recent_games = scraped
        chart_source = "KBO GraphDaily"
    else:
        st.caption(
            "KBO 자동 반영에 실패해 **로컬 CSV**로 표시합니다. "
            "(Chrome·네트워크 확인, 또는 옵션을 끄세요.)"
        )

st.caption(
    f"출처: **{chart_source}** · 기준일(그날 0시 **이전** 경기만): **{game_date}**"
)

if recent_games.empty:
    chart_df = pd.DataFrame(
        columns=["경기날짜", "홈팀", "방문팀", "관중수", "경기정보"]
    )
    st.info(
        "이 구장·기준일 이전에 표시할 과거 경기가 없습니다. "
        "경기 날짜를 늦추거나 구장을 바꿔 보세요."
    )
else:
    chart_df = recent_games[
        ["경기날짜", "홈팀", "방문팀", "관중수"]
    ].copy()
    chart_df["경기정보"] = (
        chart_df["경기날짜"].dt.strftime("%m/%d")
        + "\n"
        + chart_df["홈팀"]
        + " vs "
        + chart_df["방문팀"]
    )

chart_df.loc[len(chart_df)] = [
    pd.NaT,
    home_team,
    away_team,
    predicted_attendance,
    "이번 경기 예측"
]

fig, ax = plt.subplots(
    figsize=(11, 4)
)

fig.patch.set_facecolor("#07111f")
ax.set_facecolor("#07111f")

bars = ax.bar(
    chart_df["경기정보"],
    chart_df["관중수"]
)

_ymax = float(pd.to_numeric(chart_df["관중수"], errors="coerce").fillna(0).max()) or 1.0
_label_dy = max(_ymax * 0.015, 200.0)

# 마지막 예측 bar 색상 변경
for i, bar in enumerate(bars):

    if i == len(bars) - 1:
        bar.set_color("#18e6ff")

    else:
        bar.set_color("#4f8cff")

# 막대 위 숫자 표시
for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + _label_dy,
        f"{int(height):,}",
        ha="center",
        color="white",
        fontsize=10
    )

ax.set_title(
    f"{stadium} 최근 5경기 관중 수 + 이번 경기 예측"
)

ax.set_ylabel("관중 수")

ax.tick_params(colors="white")

ax.yaxis.label.set_color("white")

ax.title.set_color("white")

for spine in ax.spines.values():
    spine.set_color("#9fb3c8")

plt.xticks(
    rotation=0,
    color="white"
)

plt.yticks(color="white")

st.pyplot(fig)
plt.close(fig)