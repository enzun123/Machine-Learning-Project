# streamlit_app.py

import hashlib
import html
import logging
import os
import sys
from pathlib import Path

# Streamlit Cloud: common.* 는 scripts/ 패키지 — third-party·common import 전에 path 등록
_SCRIPTS = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common.congestion_levels import classify_congestion_pct
from common.kma_vilage_fcst import redact_api_secrets
from common.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

_NANUM_GOTHIC_URL = (
    "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
)


def _is_streamlit_cloud() -> bool:
    return os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "").strip().lower() == "cloud"


def _default_web_recent_enabled() -> bool:
    env = os.environ.get("STREAMLIT_WEB_RECENT")
    if env is not None:
        return env.strip() != "0"
    if _is_streamlit_cloud():
        return False
    return True


@st.cache_resource
def _ensure_korean_matplotlib_font() -> str:
    """Linux(Streamlit Cloud) 등에서 한글 깨짐 방지 — Nanum Gothic 등록."""
    plt.rcParams["axes.unicode_minus"] = False
    fonts_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    font_path = fonts_dir / "NanumGothic.ttf"

    if not font_path.is_file():
        try:
            import urllib.request

            urllib.request.urlretrieve(_NANUM_GOTHIC_URL, font_path)
        except Exception as e:
            logger.warning("Nanum Gothic 다운로드 실패: %s", e)

    if font_path.is_file():
        try:
            fm.fontManager.addfont(str(font_path))
            family = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = family
            return family
        except Exception as e:
            logger.warning("Nanum Gothic 등록 실패: %s", e)

    _font_names = {f.name for f in fm.fontManager.ttflist}
    for _fam in (
        "NanumGothic",
        "Nanum Gothic",
        "AppleGothic",
        "Apple SD Gothic Neo",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ):
        if _fam in _font_names:
            plt.rcParams["font.family"] = _fam
            return _fam

    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="KBO 관람 수요 예측 시스템",
    layout="wide",
)

_ensure_korean_matplotlib_font()

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


def _fallback_stadium_capacity() -> dict[str, int]:
    """CSV를 못 읽을 때만 사용 (kbo_stadium_info.csv와 동기 유지 권장)."""
    return {
        "잠실": 23750,
        "고척": 16744,
        "인천": 23000,
        "문학": 23000,
        "사직": 22669,
        "부산": 22669,
        "창원": 22112,
        "울산": 15000,
        "청주": 10500,
        "포항": 15000,
        "광주": 22000,
        "대전": 17000,
        "한밭": 17000,
        "대구": 24000,
        "수원": 18700,
    }


def _load_app_capacity_map() -> dict[str, int]:
    from common.stadium_capacity import load_capacity_map_for_app

    out = load_capacity_map_for_app(PROJECT_ROOT)
    return out if out else _fallback_stadium_capacity()


def _ensure_session_data() -> None:
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    # kbo_stadium_info / kbo_train_ready 갱신 시 정원 반영 (세션에 예전 값 고정 방지)
    st.session_state.cap_by_stadium = _load_app_capacity_map()


def get_capacity(stadium_name: str, home_team: str | None = None) -> int:
    from common.stadium_capacity import venue_clip_capacity

    cap_by = st.session_state.get("cap_by_stadium") or _load_app_capacity_map()
    home = home_team if home_team is not None else st.session_state.get("fld_home_team", "")
    return venue_clip_capacity(str(stadium_name), str(home), cap_by)


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
    except Exception as e:
        logger.warning(
            "KBO 최근 경기 스크랩 실패 (stadium=%s, before=%s): %s",
            stadium,
            before_iso,
            e,
            exc_info=True,
        )
        return pd.DataFrame()


def _weather_for_historical_row(attendance_df: pd.DataFrame, row: pd.Series) -> tuple[float, float, float, float]:
    """과거 경기 행에 맞는 기상(로컬 CSV 병합, 없으면 기본값)."""
    from common.config import (
        DEFAULT_RH_MEDIAN_FALLBACK,
        DEFAULT_TEMP_MEDIAN_FALLBACK,
        DEFAULT_WIND_MEDIAN_FALLBACK,
    )

    temp = DEFAULT_TEMP_MEDIAN_FALLBACK
    rain = 0.0
    hum = DEFAULT_RH_MEDIAN_FALLBACK
    wind = DEFAULT_WIND_MEDIAN_FALLBACK
    if attendance_df is None or attendance_df.empty:
        return temp, rain, hum, wind

    gdt = pd.Timestamp(row["경기날짜"]).normalize()
    home = str(row["홈팀"]).strip()
    away = str(row["방문팀"]).strip()
    st_col = str(row.get("구장", "")).strip()
    sub = attendance_df.copy()
    sub["_gd"] = pd.to_datetime(sub["경기날짜"], errors="coerce").dt.normalize()
    mask = (
        (sub["_gd"] == gdt)
        & (sub["홈팀"].astype(str).str.strip() == home)
        & (sub["방문팀"].astype(str).str.strip() == away)
    )
    if st_col:
        mask &= sub["구장"].astype(str).str.strip() == st_col
    hit = sub.loc[mask]
    if hit.empty:
        return temp, rain, hum, wind
    r = hit.iloc[-1]

    def _f(col: str, default: float) -> float:
        if col in r.index and pd.notna(r[col]):
            return float(r[col])
        return default

    return (
        _f("일평균기온(°C)", temp),
        _f("일합계강수량(mm)", rain),
        _f("일평균상대습도(%)", hum),
        _f("일평균풍속(m/s)", wind),
    )


def _predict_ml_attendance_for_row(
    tr: pd.DataFrame,
    *,
    home: str,
    away: str,
    stadium_name: str,
    game_date,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
    wind_mps: float,
    cap_map: dict[str, int],
    ml_choice_map: dict[str, bool],
) -> int | None:
    """단일 과거 경기 ML 예측(선택 모델 평균)."""
    from common.stadium_capacity import venue_clip_capacity
    from modeling.batch_feature_builder import build_ml_feature_dataframe

    cap_hi = max(1, venue_clip_capacity(stadium_name, home, cap_map))
    preds: list[int] = []
    try:
        X = build_ml_feature_dataframe(
            tr,
            home=home,
            away=away,
            stadium=stadium_name,
            game_date=game_date,
            temp_c=temp_c,
            rain_mm=rain_mm,
            hum_pct=hum_pct,
            cap_map=cap_map,
            wind_mps=wind_mps,
        )
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].astype(str).fillna("missing")
        for key, _label, fname in ML_MODEL_REGISTRY:
            if not ml_choice_map.get(key):
                continue
            pipe = _load_ml_pipeline(fname)
            if pipe is None:
                continue
            raw = float(np.asarray(pipe.predict(X))[0])
            preds.append(min(int(round(max(0.0, raw))), cap_hi))
    except Exception as e:
        logger.warning("과거 경기 ML 예측 실패 (%s vs %s): %s", home, away, e)
        return None
    if not preds:
        return None
    return int(round(float(np.mean(preds))))


def _plot_recent_actual_vs_predicted(
    compare: pd.DataFrame,
    *,
    future_pred: int,
    stadium_name: str,
) -> None:
    """최근 N경기 실제·예측 막대 비교 + 이번 경기 예측 1막대."""
    n = len(compare)
    fig, ax = plt.subplots(figsize=(max(9, n * 1.6 + 2), 4.2))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")

    labels = compare["경기"].tolist() + ["이번 경기\n(예측)"]
    x = np.arange(len(labels))
    bar_w = 0.36

    actuals = compare["실제"].to_numpy(dtype=float)
    preds = compare["예측"].to_numpy(dtype=float)

    ax.bar(
        x[:n] - bar_w / 2,
        actuals,
        bar_w,
        label="실제(기록실)",
        color="#4f8cff",
    )
    ax.bar(
        x[:n] + bar_w / 2,
        preds,
        bar_w,
        label="예측(ML)",
        color="#f59e0b",
    )
    ax.bar(
        x[n],
        future_pred,
        bar_w * 1.6,
        label="이번 경기 예측",
        color="#18e6ff",
    )

    ymax = max(
        float(np.nanmax(actuals)) if len(actuals) else 0.0,
        float(np.nanmax(preds)) if len(preds) else 0.0,
        float(future_pred),
        1.0,
    )
    dy = max(ymax * 0.015, 200.0)
    for xi, val in zip(x[:n] - bar_w / 2, actuals):
        ax.text(xi, val + dy, f"{int(val):,}", ha="center", color="white", fontsize=9)
    for xi, val in zip(x[:n] + bar_w / 2, preds):
        ax.text(xi, val + dy, f"{int(val):,}", ha="center", color="#ffe8c8", fontsize=9)
    ax.text(x[n], future_pred + dy, f"{int(future_pred):,}", ha="center", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white")
    ax.set_ylabel("관중 수")
    ax.set_title(f"{stadium_name} 최근 {n}경기 실제 vs 예측 + 이번 경기")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        fontsize=9,
        frameon=True,
        facecolor="#0d1a2b",
        edgecolor="#9fb3c8",
        labelcolor="white",
    )
    for spine in ax.spines.values():
        spine.set_color("#9fb3c8")
    plt.yticks(color="white")
    fig.subplots_adjust(bottom=0.28)
    st.pyplot(fig)
    plt.close(fig)


def _plot_ml_predictions_bar(ml_predictions: dict[str, int]) -> None:
    """알고리즘별 예측 관중 — 피처 중요도 차트와 동일한 가로 막대 스타일."""
    if not ml_predictions:
        return
    s = pd.Series({k: int(v) for k, v in ml_predictions.items()}).sort_values(ascending=True)
    labels = list(s.index)
    vals = s.to_numpy(dtype=float)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, max(2.0, 0.35 * n)))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")
    y = np.arange(n)
    bars = ax.barh(y, vals, color="#4f8cff", height=0.65)
    xmax = float(vals.max()) if len(vals) else 1.0
    dx = max(xmax * 0.015, 180.0)
    for bar in bars:
        w = float(bar.get_width())
        ax.text(
            w + dx,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}명",
            va="center",
            ha="left",
            color="#e8eef5",
            fontsize=10,
            fontweight="medium",
        )
    ax.set_xlim(0, xmax * 1.22)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#e8eef5", fontsize=10)
    ax.set_xlabel("예측 관중(명)", color="#9fb3c8", fontsize=11)
    ax.tick_params(axis="x", colors="#9fb3c8")
    ax.set_title("알고리즘별 예측 비교", color="white", fontsize=13)
    for spine in ax.spines.values():
        spine.set_color("#24384f")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


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


def _plot_rf_importance_barh(
    imp: pd.Series, top_n: int = 15, *, model_label: str = "RandomForest"
) -> plt.Figure:
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
        f"{model_label} 피처 중요도 (날씨 세부는 2그룹으로 합산)",
        color="white",
        fontsize=13,
    )
    for spine in ax.spines.values():
        spine.set_color("#24384f")
    fig.tight_layout()
    return fig


# (key, 표시 이름, joblib 파일명) — benchmark_models.py 와 동일
ML_MODEL_REGISTRY: list[tuple[str, str, str]] = [
    ("rf", "RandomForest", "attendance_rf_pipeline.joblib"),
    ("lgbm", "LightGBM", "attendance_lgbm_pipeline.joblib"),
    ("xgb", "XGBoost", "attendance_xgb_pipeline.joblib"),
]


@st.cache_resource
def _load_ml_pipeline(model_filename: str):
    p = PROJECT_ROOT / "models" / model_filename
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource
def _load_rf_pipeline():
    return _load_ml_pipeline("attendance_rf_pipeline.joblib")


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
_ensure_session_data()
df = st.session_state.df

_app_css_path = Path(__file__).resolve().parent / "styles" / "app.css"
if _app_css_path.exists():
    st.markdown(
        f"<style>{_app_css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "앱 스타일 파일이 없습니다: `scripts/app/styles/app.css`",
        icon="⚠️",
    )

# =========================
# 사이드바
# =========================
st.sidebar.title("⚾ KBO Attendance Predictor")

_ml_train_path = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
_ml_train_ok = _ml_train_path.exists()


def _ml_model_available(fname: str) -> bool:
    return (PROJECT_ROOT / "models" / fname).is_file() and _ml_train_ok


_app_mode = st.sidebar.radio(
    "작업 모드",
    ["단일 경기 예측", "미래 경기 관중수 (CSV)"],
    help="CSV 모드: 예정·미래 경기 일정 CSV → 경기별 예상 관중수를 한 번에 계산합니다.",
)

if _app_mode == "미래 경기 관중수 (CSV)":
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "일정 CSV에 **기온·강수·습도** 열이 없으면 아래 값을 모든 경기에 적용합니다."
    )
    _batch_temp = st.sidebar.slider("예상 기온(℃)", -10, 40, 18)
    _batch_rain = st.sidebar.slider(
        "일 합계 강수(mm)",
        0.0,
        120.0,
        0.0,
        0.5,
    )
    _batch_hum = st.sidebar.slider("예상 습도(%)", 0, 100, 55)

    st.sidebar.markdown("**ML 알고리즘**")
    _batch_ml_help = (
        "켜면 업로드한 일정 각 경기에 학습된 파이프라인으로 관중을 추정합니다. "
        "여러 개를 켜면 **경기별 예측은 평균**으로 표시합니다."
    )
    _batch_use_rf = st.sidebar.checkbox(
        "RandomForest",
        value=_ml_model_available("attendance_rf_pipeline.joblib"),
        disabled=not _ml_model_available("attendance_rf_pipeline.joblib"),
        help=_batch_ml_help,
    )
    _batch_use_lgbm = st.sidebar.checkbox(
        "LightGBM",
        value=False,
        disabled=not _ml_model_available("attendance_lgbm_pipeline.joblib"),
        help=_batch_ml_help,
    )
    _batch_use_xgb = st.sidebar.checkbox(
        "XGBoost",
        value=False,
        disabled=not _ml_model_available("attendance_xgb_pipeline.joblib"),
        help=_batch_ml_help,
    )
    _batch_chosen: list[str] = []
    if _batch_use_rf:
        _batch_chosen.append("RandomForest")
    if _batch_use_lgbm:
        _batch_chosen.append("LightGBM")
    if _batch_use_xgb:
        _batch_chosen.append("XGBoost")
    if not _batch_chosen and _ml_train_ok:
        st.sidebar.warning("예측 모델을 하나 이상 선택하세요.")
    if not _ml_train_ok:
        st.sidebar.caption("ML: `kbo_train_ready.csv` 없음 — `build_features.py` 실행 필요.")

    from app.csv_batch_predict_ui import render_csv_batch_predict_ui

    render_csv_batch_predict_ui(
        chosen=_batch_chosen,
        default_temp=float(_batch_temp),
        default_rain=float(_batch_rain),
        default_hum=float(_batch_hum),
        cap_by_stadium=st.session_state.cap_by_stadium,
        ml_train_ok=_ml_train_ok,
    )
    st.stop()

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
    value=_default_web_recent_enabled(),
    help=(
        "선택한 경기 날짜 이전에 치른 직전 5경기를 GraphDaily에서 가져옵니다. "
        "Chrome·Selenium 필요(Streamlit Cloud에서는 기본 꺼짐). 꺼두면 로컬 CSV만 사용합니다. "
        f"캐시 TTL {_KBO_RECENT_TTL}초."
    ),
)

st.sidebar.markdown("---")

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

wind_speed = st.sidebar.slider(
    "예상 풍속(m/s)",
    0.0,
    15.0,
    2.0,
    0.1,
    help="ML 모델의 `wind_bucket`에 반영됩니다 (학습·추론 동일 구간).",
)

st.sidebar.markdown("**ML 알고리즘**")
_ml_help = (
    "켜면 사이드바 입력(날짜·기온·습도·강수·정원 등)으로 해당 모델이 관중을 추정합니다. "
    "여러 개를 켜면 **예측값은 평균**으로 표시하고, 아래에서 알고리즘별 수치를 비교할 수 있습니다. "
    "모델 파일은 `scripts/modeling/benchmark_models.py` 로 생성합니다."
)
use_ml_rf = st.sidebar.checkbox(
    "RandomForest",
    value=_ml_model_available("attendance_rf_pipeline.joblib"),
    disabled=not _ml_model_available("attendance_rf_pipeline.joblib"),
    help=_ml_help,
)
use_ml_lgbm = st.sidebar.checkbox(
    "LightGBM",
    value=False,
    disabled=not _ml_model_available("attendance_lgbm_pipeline.joblib"),
    help=_ml_help,
)
use_ml_xgb = st.sidebar.checkbox(
    "XGBoost",
    value=False,
    disabled=not _ml_model_available("attendance_xgb_pipeline.joblib"),
    help=_ml_help,
)
use_ml_models = use_ml_rf or use_ml_lgbm or use_ml_xgb
_ml_choice_map = {
    "rf": use_ml_rf,
    "lgbm": use_ml_lgbm,
    "xgb": use_ml_xgb,
}
if use_ml_models and not _ml_train_ok:
    st.sidebar.caption("ML: `kbo_train_ready.csv` 없음 → 휴리스틱만 사용됩니다.")
_missing = [
    label
    for key, label, fname in ML_MODEL_REGISTRY
    if _ml_choice_map.get(key) and not _ml_model_available(fname)
]
if _missing:
    st.sidebar.caption(f"파일 없음(학습 필요): {', '.join(_missing)}")

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

stadium_capacity = get_capacity(stadium, home_team)

predicted_heuristic = min(
    predicted_heuristic,
    max(1, int(stadium_capacity)),
)

ml_used = False
ml_row_snapshot: dict[str, object] | None = None
ml_predictions: dict[str, int] = {}
predicted_attendance = predicted_heuristic

if use_ml_models and _ml_train_ok:

    tr = _load_kbo_train_ready()

    if tr is not None:

        try:

            from modeling.batch_feature_builder import build_ml_feature_dataframe

            cap_map = st.session_state.get("cap_by_stadium") or _load_app_capacity_map()
            X = build_ml_feature_dataframe(
                tr,
                home=home_team,
                away=away_team,
                stadium=stadium,
                game_date=game_date,
                temp_c=float(temperature),
                rain_mm=float(rainfall_mm),
                hum_pct=float(humidity),
                cap_map=cap_map,
                wind_mps=float(wind_speed),
            )
            for col in X.columns:

                if X[col].dtype == object:

                    X[col] = X[col].astype(str).fillna("missing")

            cap_hi = max(1, int(stadium_capacity))
            for key, label, fname in ML_MODEL_REGISTRY:
                if not _ml_choice_map.get(key):
                    continue
                pipe = _load_ml_pipeline(fname)
                if pipe is None:
                    continue
                raw_ml = float(np.asarray(pipe.predict(X))[0])
                pred_i = int(round(max(0.0, raw_ml)))
                ml_predictions[label] = min(pred_i, cap_hi)

            if ml_predictions:
                predicted_attendance = int(round(float(np.mean(list(ml_predictions.values())))))
                ml_used = True
                ml_row_snapshot = _ml_prediction_snapshot(
                    X.iloc[0],
                    float(temperature),
                    float(rainfall_mm),
                    float(humidity),
                )

        except Exception as e:

            logger.warning("ML 파이프라인 예측 실패, 휴리스틱 유지: %s", e, exc_info=True)
            st.caption("모델 예측에 실패해 **과거 패턴 기반 추정(휴리스틱)** 값을 표시합니다.")

if ml_used:

    _ml_labels = ", ".join(ml_predictions.keys())
    if len(ml_predictions) > 1:
        st.caption(f"예측에 **{_ml_labels}** 모델을 반영했습니다 (표시값 = 알고리즘별 예측 **평균**).")
    else:
        st.caption(f"예측에 **{_ml_labels}** 모델을 반영했습니다.")

    if len(ml_predictions) > 1:
        with st.expander("알고리즘별 예측 비교", expanded=True):
            _plot_ml_predictions_bar(ml_predictions)

    _imp_options = list(ml_predictions.keys())

    with st.expander("피처 중요도 · 이번 입력 요약", expanded=False):
        if len(_imp_options) > 1:
            _imp_model_label = st.selectbox(
                "중요도를 볼 알고리즘",
                options=_imp_options,
                key="ml_feat_imp_model",
            )
        else:
            _imp_model_label = _imp_options[0]

        st.markdown(
            f"아래 **막대 그래프**는 **{_imp_model_label}** 학습 결과에서 "
            "전체적으로 분할·분기에 자주 쓰인 변수입니다. "
            "강수·기온·습도·풍 세부 피처는 중요도 표에서 **두 줄(날씨 그룹)** 로 합산했습니다. "
            f"지금 화면의 **{predicted_attendance:,}명** 같은 **한 건의 예측**을 인과적으로 쪼개는 값(SHAP 등)은 아니며, "
            "모델이 전반적으로 어떤 정보에 무게를 두었는지 참고용입니다."
        )
        _imp_fname = next(
            (fname for _k, label, fname in ML_MODEL_REGISTRY if label == _imp_model_label),
            None,
        )
        _imp = pd.Series(dtype=float)
        if _imp_fname is not None:
            _imp_path = PROJECT_ROOT / "models" / _imp_fname
            try:
                _mt = int(os.path.getmtime(_imp_path))
            except OSError:
                _mt = 0
            _imp = _cached_rf_feature_importance_series(str(_imp_path), _mt)

        if len(_imp) > 0:
            _imp_disp = _group_rf_importance_for_display(_imp)
            _fig_imp = _plot_rf_importance_barh(
                _imp_disp, top_n=15, model_label=_imp_model_label
            )
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

    if use_ml_models and not ml_used:

        st.caption(
            "ML 알고리즘을 켰지만 필요한 모델·데이터 파일이 없어 **휴리스틱만** 사용했습니다. "
            "`benchmark_models.py` 실행 또는 체크박스 도움말(?)을 확인하세요."
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
_plan = classify_congestion_pct(congestion)
level = _plan.level
action_class = _plan.action_class
action_title = _plan.action_title
action_msg = _plan.action_msg

# =========================
# 메인 화면
# =========================
st.markdown(
    '<div class="main-title">📈 KBO 관람 수요 예측 시스템</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">'
    "사이드바에서 <b>ML 알고리즘</b>(RandomForest / LightGBM / XGBoost)을 켜면 "
    "학습된 파이프라인으로 예측하고, 끄면 <b>과거 CSV 평균 + 날씨 룰</b>만 사용합니다. "
    "최근 5경기 차트는 옵션에 따라 KBO 기록실에서 갱신할 수 있습니다."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown("## 📅 경기 정보")

st.markdown(
    f'<div class="match-text">'
    f'{html.escape(str(game_date))} | '
    f'{html.escape(str(home_team))} vs {html.escape(str(away_team))} | '
    f'{html.escape(str(stadium))}'
    f'</div>',
    unsafe_allow_html=True,
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
_gs_s = html.escape(_gs_fc.strftime("%Y-%m-%d %H:%M"))
_t3_s = html.escape(_t3.strftime("%Y-%m-%d %H:%M"))
st.markdown(
    f"""
<div class="rain-info-compact">
<b>☂️ 우천·관중 안내</b> (관중 예측과 별개 · 참고용)
<ul>
<li>관중 수는 <b>정상 개최</b> 가정. 우천 취소·노게임은 반영하지 않음.</li>
<li>언론상 검토: 개시 <b>3시간 전</b> 강우 예보 · 개시 <b>1시간 전</b> 실제 강우 (실제와 다를 수 있음).</li>
<li>기상청 동네예보: 개시 3시간 전 <b>RN1</b>, 없으면 <b>POP</b> (API 키 필요).</li>
</ul>
개시 <b>{_gs_s}</b> (KST) · 참고 <b>{_t3_s}</b>
</div>
""",
    unsafe_allow_html=True,
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
        {
            k: _ref.get(k)
            for k in (
                "ok",
                "mode",
                "base",
                "nx",
                "ny",
                "detail",
                "msg",
                "target_kst",
                "pop_anchor_kst",
            )
        },
    )
    with st.expander("기상·API 디버그 (`STREAMLIT_DEBUG_WEATHER=1`)", expanded=False):
        st.json(_ref)

if not _ref.get("ok"):
    _m = str(_ref.get("msg") or "예보 없음")
    _d = _ref.get("detail")
    _d_s = str(_d).strip() if _d is not None else ""
    _hint = html.escape(redact_api_secrets(_d_s)) if _d_s else ""
    st.markdown(
        f'<div class="rain-fcst-warn">'
        f'<div class="rain-fcst-warn-title">동네예보</div>'
        f'<p style="margin:0;">{html.escape(_m)}'
        + (f" <span class=\"rain-fcst-warn-tech\">({_hint})</span>" if _hint else "")
        + "</p></div>",
        unsafe_allow_html=True,
    )
else:
    _gout = kv.rainout_cancel_guidance(_ref)
    _gband = str(_gout["band"])
    _wrap_g = "rain-fcst-warn" if _gband == "warn" else f"rain-risk-box rain-risk-{_gband}"
    _head = html.escape(str(_gout["headline"]))
    _detail = ""
    if _ref.get("mode") == "ultra_rn1":
        mm = float(_ref["mm_h"])
        _detail = f"RN1 ≈ {mm:g} mm/h"
    elif _ref.get("mode") == "vilage_pop":
        _detail = f"POP ≈ {float(_ref['pop_pct']):.0f}%"
    _line = ""
    if _gout.get("lines"):
        _line = html.escape(
            redact_api_secrets(str(_gout["lines"][0]).replace("**", ""))
        )
    st.markdown(
        f'<div class="{_wrap_g}">'
        f'<div class="rain-risk-title">{_head}</div>'
        f'<p style="margin:6px 0 0 0;">'
        + (f"<b>{html.escape(_detail)}</b> · " if _detail else "")
        + (_line if _line else "비공식 참고 · 확정 아님")
        + "</p></div>",
        unsafe_allow_html=True,
    )

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
<div class="{html.escape(action_class)}">

<h2>{html.escape(action_title)}</h2>

<p style="font-size:18px; line-height:1.7;">
{html.escape(action_msg)}
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
    st.info(
        "이 구장·기준일 이전에 표시할 과거 경기가 없습니다. "
        "경기 날짜를 늦추거나 구장을 바꿔 보세요."
    )
else:
    hist = recent_games[["경기날짜", "홈팀", "방문팀", "관중수"]].copy()
    if "구장" in recent_games.columns:
        hist["구장"] = recent_games["구장"]
    hist["경기날짜"] = pd.to_datetime(hist["경기날짜"], errors="coerce")
    hist["관중수"] = pd.to_numeric(hist["관중수"], errors="coerce")

    compare_rows: list[dict] = []
    tr_chart = _load_kbo_train_ready() if (use_ml_models and _ml_train_ok) else None
    cap_map_chart = st.session_state.get("cap_by_stadium") or _load_app_capacity_map()

    if tr_chart is not None:
        for _, r in hist.iterrows():
            if pd.isna(r["경기날짜"]) or pd.isna(r["관중수"]):
                continue
            gdt = pd.Timestamp(r["경기날짜"])
            home_h = str(r["홈팀"]).strip()
            away_h = str(r["방문팀"]).strip()
            st_h = str(r.get("구장", stadium)).strip() or stadium
            actual_i = int(r["관중수"])
            temp, rain, hum, wind = _weather_for_historical_row(df, r)
            pred_i = _predict_ml_attendance_for_row(
                tr_chart,
                home=home_h,
                away=away_h,
                stadium_name=st_h,
                game_date=gdt,
                temp_c=temp,
                rain_mm=rain,
                hum_pct=hum,
                wind_mps=wind,
                cap_map=cap_map_chart,
                ml_choice_map=_ml_choice_map,
            )
            if pred_i is None:
                continue
            compare_rows.append(
                {
                    "경기": f"{gdt.strftime('%m/%d')}\n{home_h} vs {away_h}",
                    "실제": actual_i,
                    "예측": pred_i,
                    "오차": abs(actual_i - pred_i),
                }
            )

    if compare_rows:
        compare_df = pd.DataFrame(compare_rows)
        _plot_recent_actual_vs_predicted(
            compare_df,
            future_pred=predicted_attendance,
            stadium_name=stadium,
        )
    else:
        st.warning(
            "ML 예측 비교를 만들지 못했습니다. 사이드바에서 **ML 알고리즘**을 켜고 "
            "`kbo_train_ready.csv`·joblib이 있는지 확인하세요."
        )
        chart_df = hist.copy()
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
            "이번 경기 예측",
        ]
        fig, ax = plt.subplots(figsize=(11, 4))
        fig.patch.set_facecolor("#07111f")
        ax.set_facecolor("#07111f")
        bars = ax.bar(chart_df["경기정보"], chart_df["관중수"])
        _ymax = float(pd.to_numeric(chart_df["관중수"], errors="coerce").fillna(0).max()) or 1.0
        _label_dy = max(_ymax * 0.015, 200.0)
        for i, bar in enumerate(bars):
            bar.set_color("#18e6ff" if i == len(bars) - 1 else "#4f8cff")
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + _label_dy,
                f"{int(h):,}",
                ha="center",
                color="white",
                fontsize=10,
            )
        ax.set_title(f"{stadium} 최근 경기 관중 + 이번 경기 예측")
        ax.set_ylabel("관중 수")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#9fb3c8")
        plt.xticks(rotation=0, color="white")
        plt.yticks(color="white")
        st.pyplot(fig)
        plt.close(fig)