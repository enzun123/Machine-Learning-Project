import os
import sys

import numpy as np
import pandas as pd

# 구장명 통일(kbo_stadium_info 조인용)
STADIUM_ALIAS = {"한밭": "대전", "문학": "인천"}

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
    "일합계강수량(mm)",
    "일평균기온(°C)",
    "일평균풍속(m/s)",
    "일평균상대습도(%)",
    "관중수",
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
    "관중수",
]


def create_features_pro(df_main, df_stadium):
    df = df_main.copy()

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
        print("-" * 50)

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
