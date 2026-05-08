# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 한글 폰트 설정 (Mac)
# =========================
plt.rcParams["font.family"] = "AppleGothic"
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

    possible_paths = [

        # 현재 폴더
        BASE_DIR / "kbo_2025_attendance_weather.csv",

        # 상위 폴더들
        BASE_DIR.parent / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent.parent / "kbo_2025_attendance_weather.csv",

        # data 폴더
        BASE_DIR.parent.parent / "data" / "kbo_2025_attendance_weather.csv",
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

# =========================
# 경기장 수용 인원
# =========================
capacity_dict = {
    "잠실": 23750,
    "고척": 16000,
    "문학": 23000,
    "사직": 22758,
    "대구": 24000,
    "광주": 20500,
    "대전": 13000,
    "창원": 22112,
    "수원": 20000
}

def get_capacity(stadium_name):

    for key, value in capacity_dict.items():

        if key in str(stadium_name):
            return value

    return 20000

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

st.sidebar.markdown("---")

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

# =========================
# 임시 예측 로직
# =========================
base_df = df[
    df["구장"] == stadium
]

if len(base_df) > 0:

    predicted_attendance = int(
        base_df["관중수"].mean()
    )

else:

    predicted_attendance = int(
        df["관중수"].mean()
    )

# 날씨 보정
if rainfall > 0:

    predicted_attendance = int(
        predicted_attendance * 0.88
    )

if temperature >= 30:

    predicted_attendance = int(
        predicted_attendance * 0.93
    )

elif 15 <= temperature <= 25:

    predicted_attendance = int(
        predicted_attendance * 1.05
    )

# =========================
# 혼잡도 계산
# =========================
stadium_capacity = get_capacity(stadium)

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
    '과거 경기 데이터, 관중 데이터, 날씨 데이터를 기반으로 '
    '다음 경기의 관중 수를 예측합니다.'
    '</div>',
    unsafe_allow_html=True
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

recent_games = df[
    df["구장"] == stadium
].dropna(
    subset=["경기날짜"]
).sort_values(
    "경기날짜",
    ascending=False
).head(5)

recent_games = recent_games.sort_values(
    "경기날짜"
)

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
        height + 300,
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

# =========================
# 안내문
# =========================
st.markdown("""
<div class="notice-box">

ⓘ 본 예측은 과거 경기 데이터와 기상 정보를 기반으로 산출된 값이며,
실제 관중 수와 차이가 있을 수 있습니다.

</div>
""", unsafe_allow_html=True)