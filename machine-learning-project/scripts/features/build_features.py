"""
build_features.py  ─  개선 이력
================================
[수정1] cold-start: 시즌 첫 경기는 전 시즌 팀 평균으로 초기화
[수정2] home_prior_mean_att 누수 구조 확인 — 기존 로직 유지(행 단위 누수 없음)
[수정3] 실제 누적 승률(home/visitor_win_rate, win_rate_diff) 피처 추가
        — kbo_standings_daily.csv 의 '승률' 컬럼 조인 (없으면 0.5 대체)
[수정4] season_progress: 홈팀 소화 경기수 / 144
[수정5] 요일 원핫 플래그: is_friday / is_saturday / is_sunday
[수정6] 홈팀-방문팀 역대 매치업 평균 관중 피처 (matchup_prior_mean_att)
[수정7] 소규모/이벤트 구장 플래그: is_small_stadium (capacity < 15,000)
[수정8] 관중수 이상치 소프트 클리핑: stadium_capacity * 1.05 초과분 제거
"""

import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

_scripts_dir = Path(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from common.stadium_aliases import STADIUM_ALIAS

RAIN_BINS = [-1.0, 0.0, 1.0, 5.0, float("inf")]
RAIN_LABELS = ["No_Rain", "Rain_0_1mm", "Rain_1_5mm", "Rain_5mm_plus"]

REQUIRED_COLS = [
    "연도",
    "월",
    "주차_ISO",
    "홈팀",
    "방문팀",
    "구장",
    "요일",
    "경기날짜",
    "일합계강수량(mm)",
    "일평균기온(°C)",
    "일평균풍속(m/s)",
    "일평균상대습도(%)",
    "관중수",
]

FORM_AND_DRAW_COLS = [
    "home_prior_mean_att",
    "visitor_prior_mean_att",
    "home_last5_mean_att",
    "visitor_last5_mean_att",
    "home_draw_pct_in_league",
    "visitor_away_draw_pct_in_league",
    "home_visitor_prior_draw_diff",
    "matchup_prior_mean_att",   # [수정6]
    "season_progress",          # [수정4]
]

MODEL_READY_COLUMNS = [
    "연도",
    "월",
    "주차_ISO",
    "홈팀",
    "방문팀",
    "구장",
    "stadium_capacity",
    "is_capacity_missing",
    "is_rain",
    "rain_bucket",
    "temp_bucket",
    "is_hot",
    "humidity_bucket",
    "wind_bucket",
    "is_weekend",
    "is_friday",            # [수정5]
    "is_saturday",          # [수정5]
    "is_sunday",            # [수정5]
    "weekday_sin",
    "weekday_cos",
    "stadium_x_rain",
    "is_small_stadium",          # [수정7]
    "is_derby",                  # 더비 매치업
    "is_season_opener",          # 트랙1
    "is_childrens_day",          # 트랙1
    "home_win_rate",             # [수정3]
    "visitor_win_rate",          # [수정3]
    "win_rate_diff",             # [수정3]
    "home_gb_to_5th",            # 트랙1
    "visitor_gb_to_5th",         # 트랙1
    "is_pennant_race",           # 트랙1
    "playoff_urgency",           # 트랙1
    "month_x_playoff_urgency",   # 트랙1
    *FORM_AND_DRAW_COLS,
    "관중수",
]

# KBO 정규시즌 총 경기 수 (season_progress 분모)
FULL_SEASON_GAMES = 144

# 소규모 구장 기준 (is_small_stadium)
SMALL_STADIUM_CAPACITY = 15_000


def _pct_rank(score: float, values: list[float]) -> float:
    arr = np.array([x for x in values if not (pd.isna(x) or np.isnan(x))], dtype=float)
    if arr.size == 0 or pd.isna(score) or np.isnan(score):
        return np.nan
    arr.sort()
    below = int(np.searchsorted(arr, score, side="left"))
    equal = int(np.searchsorted(arr, score, side="right")) - below
    return (below + 0.5 * equal) / float(arr.size)


def add_season_form_and_draw_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    해당 경기 이전까지의 정보만 사용(행 단위 관중수 누수 없음).

    [수정1] cold-start: 시즌 첫 경기 → 전 시즌 팀 평균으로 초기화
    [수정4] season_progress: 홈팀 소화 경기수 / FULL_SEASON_GAMES
    [수정6] matchup_prior_mean_att: (홈팀, 방문팀) 역대 평균 관중
    """
    out = df.copy()
    if "game_no" not in out.columns:
        out["game_no"] = 1
    out["game_no"] = pd.to_numeric(out["game_no"], errors="coerce").fillna(1).astype(int)

    work = out.sort_values(["연도", "경기날짜", "game_no"], kind="mergesort").copy()
    orig_index = work.index.to_numpy()
    work = work.reset_index(drop=True)

    # [수정1] 전 시즌 팀별 평균 사전 계산 (당해 연도 아닌 전 시즌 전체 → 미래 누수 없음)
    prev_home_mean: dict[tuple[int, str], float] = {}
    prev_away_mean: dict[tuple[int, str], float] = {}
    for _y, grp in work.groupby("연도", observed=True):
        y_int = int(_y)
        for t, val in grp.groupby("홈팀", observed=True)["관중수"].mean().items():
            prev_home_mean[(y_int, str(t))] = float(val)
        for t, val in grp.groupby("방문팀", observed=True)["관중수"].mean().items():
            prev_away_mean[(y_int, str(t))] = float(val)

    teams = sorted(
        set(work["홈팀"].astype(str).unique()) | set(work["방문팀"].astype(str).unique())
    )
    hs: dict[tuple[int, str], float] = defaultdict(float)
    hc: dict[tuple[int, str], int] = defaultdict(int)
    vs: dict[tuple[int, str], float] = defaultdict(float)
    vc: dict[tuple[int, str], int] = defaultdict(int)
    home_hist: dict[tuple[int, str], deque[float]] = defaultdict(lambda: deque(maxlen=5))
    away_hist: dict[tuple[int, str], deque[float]] = defaultdict(lambda: deque(maxlen=5))
    mu_s: dict[tuple[str, str], float] = defaultdict(float)   # [수정6] matchup 누적합
    mu_c: dict[tuple[str, str], int] = defaultdict(int)        # [수정6] matchup 누적수
    team_games: dict[tuple[int, str], int] = defaultdict(int)  # [수정4] 소화 경기수

    rec: list[tuple] = []
    for j in range(len(work)):
        row = work.iloc[j]
        y = int(row["연도"])
        h = str(row["홈팀"])
        v = str(row["방문팀"])
        att = float(row["관중수"])
        kh, kv = (y, h), (y, v)

        # [수정1] cold-start → 전 시즌 팀 평균
        if hc[kh] > 0:
            h_prior = hs[kh] / hc[kh]
        else:
            h_prior = prev_home_mean.get((y - 1, h), np.nan)

        if vc[kv] > 0:
            v_prior = vs[kv] / vc[kv]
        else:
            v_prior = prev_away_mean.get((y - 1, v), np.nan)

        h5 = (
            float(np.mean(home_hist[kh]))
            if home_hist[kh]
            else (h_prior if pd.notna(h_prior) else np.nan)
        )
        v5 = (
            float(np.mean(away_hist[kv]))
            if away_hist[kv]
            else (v_prior if pd.notna(v_prior) else np.nan)
        )

        league_home = [hs[(y, t)] / hc[(y, t)] for t in teams if hc[(y, t)] > 0]
        league_away = [vs[(y, t)] / vc[(y, t)] for t in teams if vc[(y, t)] > 0]
        h_pct = (
            _pct_rank(float(h_prior), league_home)
            if (hc[kh] > 0 and pd.notna(h_prior))
            else np.nan
        )
        v_pct = (
            _pct_rank(float(v_prior), league_away)
            if (vc[kv] > 0 and pd.notna(v_prior))
            else np.nan
        )
        h_diff = (
            float(h_prior - v_prior)
            if not (pd.isna(h_prior) or pd.isna(v_prior))
            else np.nan
        )

        # [수정6] matchup prior
        mu_key = (h, v)
        mu_prior = mu_s[mu_key] / mu_c[mu_key] if mu_c[mu_key] > 0 else np.nan

        # [수정4] season_progress
        h_progress = team_games[kh] / FULL_SEASON_GAMES

        rec.append(
            (h_prior, v_prior, h5, v5, h_pct, v_pct, h_diff, mu_prior, h_progress)
        )

        hs[kh] += att
        hc[kh] += 1
        vs[kv] += att
        vc[kv] += 1
        home_hist[kh].append(att)
        away_hist[kv].append(att)
        mu_s[mu_key] += att
        mu_c[mu_key] += 1
        team_games[kh] += 1
        team_games[kv] += 1

    feat = pd.DataFrame(rec, columns=FORM_AND_DRAW_COLS)
    work[FORM_AND_DRAW_COLS] = feat.values
    work.index = orig_index
    joined = out.join(work[FORM_AND_DRAW_COLS], how="left")
    for c in FORM_AND_DRAW_COLS:
        med_y = joined.groupby("연도", observed=True)[c].transform("median")
        joined[c] = joined[c].fillna(med_y).fillna(joined[c].median())
        joined[c] = joined[c].fillna(0.0)
    return joined


def join_standings_features(
    df: pd.DataFrame, standings_path: str | Path | None
) -> pd.DataFrame:
    """
    [수정3 확장] kbo_standings_daily.csv 에서 경기 당일 기준 조인.

    - home/visitor_win_rate   : 당일 실제 승률
    - win_rate_diff           : 홈-방문 승률 차
    - home/visitor_gb_to_5th  : 5위와의 게임 차 (양수=5위 밖, 음수=5위 안)
    - is_pennant_race         : 홈팀 8월↑ & 5위와 게임 차 ≤3 플래그
    """
    out = df.copy()
    p = Path(standings_path) if standings_path else None
    default_wr = 0.5

    if p is None or not p.exists():
        for col, val in {
            "home_win_rate": default_wr, "visitor_win_rate": default_wr,
            "win_rate_diff": 0.0,
            "home_gb_to_5th": 0.0, "visitor_gb_to_5th": 0.0,
            "is_pennant_race": 0,
        }.items():
            out[col] = val
        return out

    st = pd.read_csv(p, encoding="utf-8-sig")
    if "시리즈" in st.columns:
        st = st[st["시리즈"].astype(str).str.contains("정규", na=False)]
    st["기준일"] = pd.to_datetime(st["기준일"], errors="coerce").dt.normalize()
    st["승률"]  = pd.to_numeric(st["승률"],  errors="coerce").fillna(default_wr)
    st["게임차"] = pd.to_numeric(st["게임차"], errors="coerce").fillna(0.0)
    st["순위"]   = pd.to_numeric(st["순위"],   errors="coerce").fillna(5)

    # 날짜별 5위 팀의 게임차 (타이 있으면 5번째 행)
    rank5_gb = (
        st.sort_values(["기준일", "순위"])
        .groupby("기준일", observed=True)
        .apply(
            lambda g: float(g.iloc[min(4, len(g) - 1)]["게임차"]),
            include_groups=False,
        )
        .reset_index(name="gb_5th")
    )
    st = st.merge(rank5_gb, on="기준일", how="left")
    st["gb_5th"] = st["gb_5th"].fillna(st["게임차"])
    st["gb_to_5th"] = st["게임차"] - st["gb_5th"]

    gdt = pd.to_datetime(out["경기날짜"], errors="coerce").dt.normalize()
    work = out.assign(_gdt=gdt)

    st_h = st[["기준일", "팀명", "승률", "gb_to_5th"]].rename(
        columns={"승률": "home_win_rate", "gb_to_5th": "home_gb_to_5th"}
    )
    work = work.merge(
        st_h, left_on=["_gdt", "홈팀"], right_on=["기준일", "팀명"], how="left"
    ).drop(columns=["기준일", "팀명"], errors="ignore")

    st_v = st[["기준일", "팀명", "승률", "gb_to_5th"]].rename(
        columns={"승률": "visitor_win_rate", "gb_to_5th": "visitor_gb_to_5th"}
    )
    work = work.merge(
        st_v, left_on=["_gdt", "방문팀"], right_on=["기준일", "팀명"], how="left"
    ).drop(columns=["기준일", "팀명", "_gdt"], errors="ignore")

    for col, default in [
        ("home_win_rate", default_wr), ("visitor_win_rate", default_wr),
        ("home_gb_to_5th", 0.0),       ("visitor_gb_to_5th", 0.0),
    ]:
        med_y = work.groupby("연도", observed=True)[col].transform("median")
        work[col] = work[col].fillna(med_y).fillna(default)
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)

    work["win_rate_diff"] = work["home_win_rate"] - work["visitor_win_rate"]

    month_num = pd.to_numeric(work["월"], errors="coerce").fillna(0)
    work["is_pennant_race"] = (
        (month_num >= 8) & (work["home_gb_to_5th"].abs() <= 3.0)
    ).astype(int)

    return work


def add_calendar_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    규칙 기반 특수 경기 플래그 (target 미사용 — 누수 없음).
    - is_season_opener : 해당 연도 홈팀의 시즌 첫 홈경기
    - is_childrens_day : 경기날짜가 5월 4·5·6일 (어린이날 전후)
    """
    out = df.copy()
    dt = pd.to_datetime(out["경기날짜"], errors="coerce")

    out["is_childrens_day"] = (
        dt.notna() & (dt.dt.month == 5) & (dt.dt.day.isin([4, 5, 6]))
    ).astype(int)

    out["_dt"] = dt
    opener = (
        out.dropna(subset=["_dt"])
        .groupby(["연도", "홈팀"], observed=True)["_dt"]
        .min()
        .reset_index()
        .rename(columns={"_dt": "_opener_dt"})
    )
    out = out.merge(opener, on=["연도", "홈팀"], how="left")
    out["is_season_opener"] = (out["_dt"] == out["_opener_dt"]).astype(int)
    return out.drop(columns=["_dt", "_opener_dt"], errors="ignore")


def create_features_pro(df_main, df_stadium, standings_path=None):
    df = df_main.copy()
    df["관중수"] = pd.to_numeric(df["관중수"], errors="coerce")

    df["구장"] = df["구장"].replace(STADIUM_ALIAS)

    for col in [
        "일합계강수량(mm)",
        "일평균기온(°C)",
        "일평균풍속(m/s)",
        "일평균상대습도(%)",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df_stadium.empty and "구장" in df_stadium.columns and "최대수용인원" in df_stadium.columns:
        st = df_stadium.copy()
        st["구장"] = st["구장"].replace(STADIUM_ALIAS)
        st["최대수용인원"] = pd.to_numeric(st["최대수용인원"], errors="coerce")
        st_max_table = st.dropna(subset=["구장", "최대수용인원"]).groupby("구장")["최대수용인원"].max()
        df["stadium_capacity"] = df["구장"].map(st_max_table)
    else:
        df["stadium_capacity"] = np.nan

    df["is_capacity_missing"] = df["stadium_capacity"].isnull().astype(int)
    cap_mean = df["stadium_capacity"].mean()
    if pd.isna(cap_mean):
        cap_mean = 0.0
    df["stadium_capacity"] = df["stadium_capacity"].fillna(cap_mean)

    # [수정8] 관중수 소프트 클리핑 (구장 정원 105% 초과는 데이터 오류로 간주)
    cap_clip = df["stadium_capacity"].clip(lower=1.0) * 1.05
    df["관중수"] = df["관중수"].clip(upper=cap_clip)

    df["일합계강수량(mm)"] = df["일합계강수량(mm)"].fillna(0)
    df["is_rain"] = (df["일합계강수량(mm)"] > 0).astype(int)

    df["rain_bucket"] = pd.cut(
        df["일합계강수량(mm)"],
        bins=RAIN_BINS,
        labels=RAIN_LABELS,
        include_lowest=True,
    )

    ta = df["일평균기온(°C)"]
    ta_med = float(ta.median()) if pd.notna(ta.median()) else 15.0
    df["일평균기온(°C)"] = ta.fillna(ta_med)
    rh = df["일평균상대습도(%)"]
    rh_med = float(rh.median()) if pd.notna(rh.median()) else 60.0
    df["일평균상대습도(%)"] = rh.fillna(rh_med).clip(0, 100)
    ws = df["일평균풍속(m/s)"]
    ws_med = float(ws.median()) if pd.notna(ws.median()) else 2.0
    df["일평균풍속(m/s)"] = ws.fillna(ws_med).clip(lower=0)

    df["is_hot"] = (df["일평균기온(°C)"] >= 30).astype(int)
    df["temp_bucket"] = pd.cut(
        df["일평균기온(°C)"],
        bins=[-float("inf"), 10, 20, 25, 30, float("inf")],
        labels=["VeryCold", "Cold", "Mild", "Warm", "Hot"],
    )

    df["humidity_bucket"] = pd.cut(
        df["일평균상대습도(%)"],
        bins=[0, 40, 60, 80, 100],
        labels=["Dry", "Normal", "Humid", "VeryHumid"],
    )
    df["wind_bucket"] = pd.cut(
        df["일평균풍속(m/s)"],
        bins=[-1, 1.5, 3.3, 5.4, float("inf")],
        labels=["Calm", "Light", "Moderate", "Strong"],
    )

    df["is_weekend"] = df["요일"].isin(["토", "일"]).astype(int)
    weekday_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
    df["weekday_num"] = df["요일"].map(weekday_map)
    valid_wd = df["weekday_num"].between(0, 6)
    df["weekday_sin"] = np.where(valid_wd, np.sin(2 * np.pi * df["weekday_num"] / 7), 0.0)
    df["weekday_cos"] = np.where(valid_wd, np.cos(2 * np.pi * df["weekday_num"] / 7), 0.0)

    # [수정5] 요일 원핫 플래그
    df["is_friday"] = (df["요일"] == "금").astype(int)
    df["is_saturday"] = (df["요일"] == "토").astype(int)
    df["is_sunday"] = (df["요일"] == "일").astype(int)

    df["stadium_x_rain"] = df["구장"].astype(str) + "_" + df["is_rain"].astype(str)

    # [수정7] 소규모/이벤트 구장 플래그
    df["is_small_stadium"] = (df["stadium_capacity"] < SMALL_STADIUM_CAPACITY).astype(int)

    # 더비 매치업 플래그 (라이벌 경기 — 홈/원정 관계없이 양방향)
    derby_pairs: set[frozenset[str]] = {
        frozenset({"LG", "두산"}),
        frozenset({"롯데", "NC"}),
        frozenset({"삼성", "KIA"}),
        frozenset({"KT", "키움"}),
        frozenset({"SSG", "한화"}),
    }
    df["is_derby"] = [
        int(frozenset({str(h), str(v)}) in derby_pairs)
        for h, v in zip(df["홈팀"].astype(str), df["방문팀"].astype(str))
    ]

    # 트랙1: 이벤트 플래그 (개막전, 어린이날)
    df = add_calendar_event_flags(df)

    # 트랙1: 승률·5위 게임차·페넌트레이스 플래그 조인
    df = join_standings_features(df, standings_path)

    # 트랙1: 월 × 포스트시즌 긴박도 교호작용
    df["playoff_urgency"] = (3.0 - df["home_gb_to_5th"].clip(-10, 3)).clip(lower=0.0)
    df["month_x_playoff_urgency"] = (
        pd.to_numeric(df["월"], errors="coerce").fillna(0) * df["playoff_urgency"]
    )

    df = add_season_form_and_draw_proxy(df)
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    path_main = os.path.join(project_root, "data/processed/final_dataset.csv")
    path_stadium = os.path.join(project_root, "data/external/kbo_stadium_info.csv")
    path_standings = os.path.join(project_root, "data/external/kbo_standings_daily.csv")
    save_path = os.path.join(project_root, "data/processed/kbo_train_ready.csv")

    try:
        if not os.path.exists(path_main):
            raise FileNotFoundError(
                f"입력 CSV 없음: {path_main}\n"
                "먼저 실행: python3 scripts/preprocessing/preprocess_attendance_weather.py"
            )

        df_final = pd.read_csv(path_main, encoding="utf-8-sig")
        df_st_info = (
            pd.read_csv(path_stadium, encoding="utf-8-sig")
            if os.path.exists(path_stadium)
            else pd.DataFrame()
        )

        missing = [c for c in REQUIRED_COLS if c not in df_final.columns]
        if missing:
            raise KeyError(f"필수 컬럼 누락: {missing}")

        processed_df = create_features_pro(df_final, df_st_info, path_standings)

        final_ready_df = processed_df[MODEL_READY_COLUMNS]
        final_ready_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print("-" * 50)
        print("Success: Final Data for ML pipeline is generated.")
        print(f"Location: {save_path}")
        print("rain_bucket: No_Rain / Rain_0_1mm / Rain_1_5mm / Rain_5mm_plus")
        print("[수정1] cold-start: 전 시즌 팀 평균으로 초기화")
        print("[수정3] win_rate: kbo_standings_daily.csv 승률 조인")
        print("[수정4] season_progress: 소화 경기수 / 144")
        print("[수정5] is_friday / is_saturday / is_sunday 플래그 추가")
        print("[수정6] matchup_prior_mean_att: 역대 매치업 평균 관중")
        print("[수정7] is_small_stadium: capacity < 15,000")
        print("[수정8] 관중수 이상치 소프트 클리핑 (정원 105% 초과 제거)")
        print("[트랙1] gb_to_5th, is_pennant_race, is_season_opener, is_childrens_day, playoff_urgency")
        print("[더비] is_derby: LG-두산, 롯데-NC, 삼성-KIA, KT-키움, SSG-한화")
        print("-" * 50)

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
