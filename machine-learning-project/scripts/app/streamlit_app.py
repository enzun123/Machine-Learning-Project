# streamlit_app.py

import logging
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

df = load_data()

_SCRIPTS = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _fallback_stadium_capacity() -> dict[str, int]:
    """CSV를 못 읽을 때만 사용 (kbo_stadium_info.csv와 동기 유지 권장)."""
    return {
        "잠실": 25000,
        "고척": 17000,
        "인천": 23000,
        "문학": 23000,
        "사직": 23500,
        "부산": 23500,
        "창원": 22000,
        "울산": 22000,
        "청주": 10500,
        "포항": 9000,
        "광주": 20500,
        "대전": 20007,
        "한밭": 13000,
        "대구": 24000,
        "수원": 20000,
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

.notice-box {
    background-color: #101d2d;
    border: 1px solid #263f5c;
    border-radius: 10px;
    padding: 15px;
    color: #b8c7d9;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 사이드바
# =========================
st.sidebar.title("⚾ KBO Attendance Predictor")

game_date = st.sidebar.date_input("경기 날짜")
st.sidebar.caption(
    "차트·KBO 최근 5경기 및 **RF 모델** 입력에 쓰는 기준일(당일 0시 이전 경기만). "
    "RF를 끈 **휴리스틱** 예측 숫자에는 이 날짜가 반영되지 않습니다."
)

home_team = st.sidebar.selectbox(
    "홈팀",
    sorted(df["홈팀"].dropna().unique())
)

away_team = st.sidebar.selectbox(
    "원정팀",
    sorted(df["방문팀"].dropna().unique())
)

stadium = st.sidebar.selectbox(
    "경기장",
    sorted(df["구장"].dropna().unique())
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

rainfall = st.sidebar.slider(
    "예상 강수량(mm)",
    0.0,
    50.0,
    0.0
)

humidity = st.sidebar.slider(
    "예상 습도(%)",
    0,
    100,
    60
)

use_rf_model = st.sidebar.checkbox(
    "RF 파이프라인 예측 (`attendance_rf_pipeline.joblib`)",
    value=_rf_artifacts_ok,
    disabled=not _rf_artifacts_ok,
    help=(
        "`kbo_train_ready.csv`에서 홈·원정·구장이 가장 가까운 과거 행을 고른 뒤, "
        "위 날짜·기온·강수·습도·정원을 덮어 RF로 예측합니다. "
        "파일이 없거나 오류 시 아래 룰 기반만 사용합니다."
    ),
)

st.sidebar.caption(
    "RF 사용 시: 학습 시와 같은 피처 범주(강수·기온·습도 버킷 등)로 넣습니다. "
    "RF를 끄면 기온·강수·습도는 **룰 기반** 추정에만 쓰입니다."
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

# 휴리스틱: 날씨 룰
if rainfall > 0:

    predicted_heuristic = int(
        predicted_heuristic * 0.88
    )

if temperature >= 30:

    predicted_heuristic = int(
        predicted_heuristic * 0.93
    )

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
                float(rainfall),
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

        except Exception:

            logger.warning("RF 파이프라인 예측 실패, 휴리스틱 유지", exc_info=True)

if ml_used:

    st.caption(
        "예측: **RandomForest 파이프라인** `models/attendance_rf_pipeline.joblib` "
        "(입력: `kbo_train_ready` 유사 행 + 사이드바 **날짜·기온·강수·습도**)."
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
            "RF 모델 파일 또는 `kbo_train_ready.csv`가 없어 **휴리스틱만** 사용합니다."
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

# =========================
# 안내문
# =========================
st.markdown("""
<div class="notice-box">

ⓘ 사이드바 <b>RF 파이프라인</b>을 켠 경우: 저장된 회귀 파이프라인으로 추정합니다. 입력은
<code>kbo_train_ready.csv</code>에서 고른 유사 행에 UI의 날짜·기상을 반영한 값입니다.
RF를 끄면 <b>휴리스틱(과거 평균·비·폭염·고습도 룰)</b>만 사용합니다.
구장 정원은 <code>kbo_stadium_info.csv</code>를 우선 사용합니다.

</div>
""", unsafe_allow_html=True)