"""
interim 관중+기상 CSV → final_dataset.csv 병합·정리 (feat/preprocessing)

입력: data/interim/kbo_*_attendance_weather.csv (weather_api 등으로 생성)
출력: data/processed/final_dataset.csv

실행:
  cd machine-learning-project
  python3 scripts/preprocessing/preprocess_attendance_weather.py
  python3 scripts/preprocessing/preprocess_attendance_weather.py --output data/processed/final_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

# build_features.py 와 동일 — 구장 수용·기상 조인 일관성
STADIUM_ALIAS = {"한밭": "대전", "문학": "인천"}

DEFAULT_INPUTS = [
    ROOT_DIR / "data" / "interim" / "kbo_2024_attendance_weather.csv",
    ROOT_DIR / "data" / "interim" / "kbo_2025_attendance_weather.csv",
]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "final_dataset.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="interim 관중+기상 → final_dataset 병합")
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=None,
        help="입력 CSV 경로들 (기본: interim kbo_2024/2025_attendance_weather.csv)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="출력 CSV 경로",
    )
    return p.parse_args()


def _parse_crowd_series(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.replace(",", "", regex=False)
    t = t.str.replace(r"[^\d]", "", regex=True).replace("", pd.NA)
    return pd.to_numeric(t, errors="coerce")


def main() -> None:
    args = _parse_args()
    input_paths = list(args.inputs) if args.inputs is not None else list(DEFAULT_INPUTS)
    output_path = Path(args.output)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        print(
            "FATAL: 다음 입력 파일이 없습니다.\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n(2단계 weather_api 생략 시, 기존 interim을 두거나 weather_api를 먼저 실행하세요.)",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in input_paths]
    df = pd.concat(frames, ignore_index=True)
    n_in = len(df)

    # 완전 동일 행만 제거합니다.
    df = df.drop_duplicates().copy()
    if len(df) < n_in:
        print(f"[검증] 완전 중복 행 제거: {n_in - len(df)}건")

    if "구장" in df.columns:
        df["구장"] = df["구장"].replace(STADIUM_ALIAS)

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
            if col == "관중수":
                df[col] = _parse_crowd_series(df[col])
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["관중수"]).copy()
    df["관중수"] = df["관중수"].astype(int)
    if len(df) < n_before:
        print(f"[검증] 관중수 결측/비숫자 행 제거: {n_before - len(df)}건")

    suspicious = (df["관중수"] > 50000) | (df["관중수"] < 0)
    if suspicious.any():
        print(f"[검증] 관중수 5만 초과 또는 음수: {int(suspicious.sum())}건 — 원본 확인 권장")

    dup_key = [c for c in ["경기날짜", "홈팀", "방문팀", "구장", "game_no"] if c in df.columns]
    if len(dup_key) >= 4 and df.duplicated(subset=dup_key).any():
        print(f"[검증] 경고: 키 {dup_key} 기준 완전 중복 행이 있습니다.")

    # 읽기 편한 순서로 정렬합니다.
    sort_cols = [c for c in ["연도", "경기날짜", "경기일시", "홈팀", "방문팀", "구장", "game_no"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    dup_count = int(df.duplicated(match_key).sum())
    rain_null = int(df[rain_col].isna().sum())
    print("-" * 50)
    print(f"saved: {output_path}")
    print(f"rows: {len(df)}, cols: {len(df.columns)}")
    print(f"inputs: {[str(p) for p in input_paths]}")
    print(f"duplicate_by_match_key(연도·경기일시·홈·방문·구장): {dup_count}")
    print(f"rain_missing_after_fill: {rain_null}")
    if "구장" in df.columns:
        print("구장별 행 수 (상위 5):")
        print(df["구장"].value_counts().head(5).to_string())
    print("-" * 50)


if __name__ == "__main__":
    main()
