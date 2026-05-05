from __future__ import annotations

import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILES = [
    ROOT_DIR / "weather_api" / "kbo_2024_attendance_weather.csv",
    ROOT_DIR / "weather_api" / "kbo_2025_attendance_weather.csv",
]
OUTPUT_FILE = ROOT_DIR / "kbo_attendance_weather_preprocessed.csv"


def main() -> None:
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in INPUT_FILES]
    df = pd.concat(frames, ignore_index=True)

    # 완전 동일 행만 제거합니다.
    df = df.drop_duplicates().copy()

    # 경기 키 중복은 더블헤더 가능성을 고려해 유지하고 game_no로 구분합니다.
    match_key = ["연도", "경기일시", "홈팀", "방문팀", "구장"]
    for col in match_key:
        if col not in df.columns:
            raise KeyError(f"필수 키 컬럼이 없습니다: {col}")
    df["game_no"] = df.groupby(match_key).cumcount() + 1

    # 강수 결측은 0으로 처리합니다.
    rain_col = "일합계강수량(mm)"
    if rain_col not in df.columns:
        raise KeyError(f"강수 컬럼이 없습니다: {rain_col}")
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0)

    # 수치형 컬럼은 숫자 타입으로 정리합니다.
    numeric_cols = ["관중수", "지점번호", "일평균기온(°C)", "일평균풍속(m/s)", "일평균상대습도(%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 읽기 편한 순서로 정렬합니다.
    sort_cols = [c for c in ["연도", "경기날짜", "경기일시", "홈팀", "방문팀", "구장", "game_no"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    dup_count = int(df.duplicated(match_key).sum())
    rain_null = int(df[rain_col].isna().sum())
    print(f"saved: {OUTPUT_FILE}")
    print(f"rows: {len(df)}, cols: {len(df.columns)}")
    print(f"duplicate_by_match_key: {dup_count}")
    print(f"rain_missing_after_fill: {rain_null}")


if __name__ == "__main__":
    main()
