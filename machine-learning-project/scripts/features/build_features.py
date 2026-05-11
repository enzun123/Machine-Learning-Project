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

# EDA 명세: {0, (0,1], (1,5], (5,inf)} — pandas read_csv 기본 NA와 겹치지 않게 No_Rain 사용
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

# 공식 순위·승패가 없을 때: 경기일시 순 ‘폼’(지난 홈/원정 관중) + 리그 내 상대 퍼센타일(인기도 순위 proxy)
FORM_AND_DRAW_COLS = [
    "home_prior_mean_att",
    "visitor_prior_mean_att",
    "home_last5_mean_att",
    "visitor_last5_mean_att",
    "home_draw_pct_in_league",
    "visitor_away_draw_pct_in_league",
    "home_visitor_prior_draw_diff",
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
    "weekday_sin",
    "weekday_cos",
    "stadium_x_rain",
    *FORM_AND_DRAW_COLS,
    "관중수",
]


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
    해당 경기 이전까지의 정보만 사용(관중수 누수 없음).

    - 폼: 홈팀·방문팀 각각 홈/원정으로 치른 과거 경기의 평균·최근 5경기 평균 관중
    - 순위 proxy: 같은 연도에서, 위 누적 평균을 리그 전체 팀과 비교한 퍼센타일(0~1)
      (공식 기록실 순위가 아님)
    """
    out = df.copy()
    if "game_no" not in out.columns:
        out["game_no"] = 1
    out["game_no"] = pd.to_numeric(out["game_no"], errors="coerce").fillna(1).astype(int)

    work = out.sort_values(["연도", "경기날짜", "game_no"], kind="mergesort").copy()
    orig_index = work.index.to_numpy()
    work = work.reset_index(drop=True)

    teams = sorted(set(work["홈팀"].astype(str).unique()) | set(work["방문팀"].astype(str).unique()))
    hs: dict[tuple[int, str], float] = defaultdict(float)
    hc: dict[tuple[int, str], int] = defaultdict(int)
    vs: dict[tuple[int, str], float] = defaultdict(float)
    vc: dict[tuple[int, str], int] = defaultdict(int)
    home_hist: dict[tuple[int, str], deque[float]] = defaultdict(lambda: deque(maxlen=5))
    away_hist: dict[tuple[int, str], deque[float]] = defaultdict(lambda: deque(maxlen=5))

    rec: list[tuple[float, float, float, float, float, float, float]] = []
    for j in range(len(work)):
        row = work.iloc[j]
        y = int(row["연도"])
        h = str(row["홈팀"])
        v = str(row["방문팀"])
        att = float(row["관중수"])
        kh, kv = (y, h), (y, v)

        h_prior = hs[kh] / hc[kh] if hc[kh] else np.nan
        v_prior = vs[kv] / vc[kv] if vc[kv] else np.nan
        h5 = float(np.mean(home_hist[kh])) if home_hist[kh] else np.nan
        v5 = float(np.mean(away_hist[kv])) if away_hist[kv] else np.nan

        league_home = [hs[(y, t)] / hc[(y, t)] for t in teams if hc[(y, t)] > 0]
        league_away = [vs[(y, t)] / vc[(y, t)] for t in teams if vc[(y, t)] > 0]

        h_pct = _pct_rank(float(h_prior), league_home) if hc[kh] > 0 else np.nan
        v_pct = _pct_rank(float(v_prior), league_away) if vc[kv] > 0 else np.nan

        if not (pd.isna(h_prior) or pd.isna(v_prior)):
            h_diff = float(h_prior - v_prior)
        else:
            h_diff = np.nan

        rec.append((float(h_prior), float(v_prior), h5, v5, h_pct, v_pct, h_diff))

        hs[kh] += att
        hc[kh] += 1
        vs[kv] += att
        vc[kv] += 1
        home_hist[kh].append(att)
        away_hist[kv].append(att)

    feat = pd.DataFrame(rec, columns=FORM_AND_DRAW_COLS)
    work[FORM_AND_DRAW_COLS] = feat.values
    work.index = orig_index
    joined = out.join(work[FORM_AND_DRAW_COLS], how="left")
    for c in FORM_AND_DRAW_COLS:
        med_y = joined.groupby("연도", observed=True)[c].transform("median")
        joined[c] = joined[c].fillna(med_y).fillna(joined[c].median())
        joined[c] = joined[c].fillna(0.0)
    return joined


def create_features_pro(df_main, df_stadium):
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

    df["일합계강수량(mm)"] = df["일합계강수량(mm)"].fillna(0)
    df["is_rain"] = (df["일합계강수량(mm)"] > 0).astype(int)

    df["rain_bucket"] = pd.cut(
        df["일합계강수량(mm)"],
        bins=RAIN_BINS,
        labels=RAIN_LABELS,
        include_lowest=True,
    )

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

    df["stadium_x_rain"] = df["구장"].astype(str) + "_" + df["is_rain"].astype(str)

    df = add_season_form_and_draw_proxy(df)
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    path_main = os.path.join(project_root, "data/processed/final_dataset.csv")
    path_stadium = os.path.join(project_root, "data/external/kbo_stadium_info.csv")
    save_path = os.path.join(project_root, "data/processed/kbo_train_ready.csv")

    try:
        if not os.path.exists(path_main):
            path_main = os.path.join(project_root, "data/processed/final_dataset (2).csv")
        if not os.path.exists(path_main):
            raise FileNotFoundError(f"입력 CSV 없음: {path_main}")

        df_final = pd.read_csv(path_main, encoding="utf-8-sig")
        df_st_info = pd.read_csv(path_stadium, encoding="utf-8-sig") if os.path.exists(path_stadium) else pd.DataFrame()

        missing = [c for c in REQUIRED_COLS if c not in df_final.columns]
        if missing:
            raise KeyError(f"필수 컬럼 누락: {missing}")

        processed_df = create_features_pro(df_final, df_st_info)

        final_ready_df = processed_df[MODEL_READY_COLUMNS]
        final_ready_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print("-" * 50)
        print("Success: Final Data for ML pipeline is generated.")
        print(f"Location: {save_path}")
        print("rain_bucket: No_Rain / Rain_0_1mm / Rain_1_5mm / Rain_5mm_plus (명세 구간)")
        print("form/rank proxy: home/visitor prior & last5 att; *_draw_pct_in_league (공식 순위 아님)")
        print("-" * 50)

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
