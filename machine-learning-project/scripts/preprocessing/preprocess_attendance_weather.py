"""
interim 관중+기상 CSV → final_dataset.csv 병합·정리 (feat/preprocessing)

입력(기본):
  data/interim/kbo_2024_attendance_weather.csv
  data/interim/kbo_2025_attendance_weather.csv
  — `scripts/data_collection/weather_api.py` 로 생성하거나, 동일 컬럼 스키마의 CSV를
    `--inputs` 로 직접 넘깁니다. (weather 단계 생략 시에도 interim 파일이 있어야 합니다.)

출력:
  data/processed/final_dataset.csv

정렬·키 (build_features 시계열 피처와 동일 계약):
  - `경기날짜`: `경기일시`에서 파생·보정(없을 때). YYYY-MM-DD 문자열.
  - `game_no`: 같은 (연도, 경기날짜, 홈팀, 방문팀, 구장) 내에서 `경기일시` 순으로 1,2,… (더블헤더).
  - 최종 행 순서: 연도 → 경기날짜 → game_no → 경기일시 → … (아래 SORT_OUTPUT 참고)

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
_scripts_dir = Path(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from common.stadium_aliases import STADIUM_ALIAS

# build_features 시계열 정렬과 맞춤. 더블헤더 순서는 `_ts_for_sort`(경기일시 파싱)로만 구분.
SORT_FOR_GAME_NO = ["연도", "경기날짜", "홈팀", "방문팀", "구장", "_ts_for_sort"]
GAME_NO_GROUP_KEY = ["연도", "경기날짜", "홈팀", "방문팀", "구장"]
SORT_OUTPUT = ["연도", "경기날짜", "game_no", "_ts_for_sort", "홈팀", "방문팀", "구장"]

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


def _ensure_경기날짜_and_ts(df: pd.DataFrame) -> pd.DataFrame:
    """경기일시 파싱 + 경기날짜(YYYY-MM-DD) 보장. build_features 정렬과 동일한 날짜 키."""
    out = df.copy()
    ts = pd.to_datetime(out["경기일시"], errors="coerce")
    if "경기날짜" not in out.columns:
        out["경기날짜"] = pd.NA
    rd = pd.to_datetime(out["경기날짜"], errors="coerce")
    # 경기날짜 비어 있으면 경기일시에서만 채움
    fill = ts.dt.normalize()
    out["경기날짜"] = rd.fillna(fill).dt.strftime("%Y-%m-%d")
    mask_bad = out["경기날짜"].isin(["NaT", "NaN", ""]) | out["경기날짜"].isna()
    if mask_bad.any():
        out.loc[mask_bad, "경기날짜"] = ts.loc[mask_bad].dt.strftime("%Y-%m-%d")
    out["_ts_for_sort"] = ts
    return out


def main() -> None:
    args = _parse_args()
    input_paths = list(args.inputs) if args.inputs is not None else list(DEFAULT_INPUTS)
    output_path = Path(args.output)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        print(
            "FATAL: 다음 입력 파일이 없습니다.\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\n해결 방법:\n"
            "  1) `data/interim/` 에 `kbo_*_attendance_weather.csv` 가 있는지 확인\n"
            "  2) 없으면 `python3 scripts/data_collection/weather_api.py` 로 생성\n"
            "     (또는 동일 스키마 CSV를 `--inputs 경로1 경로2` 로 지정)\n"
            "  3) 기본 interim 경로:\n"
            + "\n".join(f"     - {p}" for p in DEFAULT_INPUTS),
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

    for col in ("경기일시", "홈팀", "방문팀", "구장", "연도"):
        if col not in df.columns:
            raise KeyError(f"필수 키 컬럼이 없습니다: {col}")

    df = _ensure_경기날짜_and_ts(df)

    # 더블헤더: 같은 (연도, 경기날짜, 홈, 방문, 구장)에서 경기일시(파싱) 순으로 game_no 부여
    sort_cols_gn = [c for c in SORT_FOR_GAME_NO if c in df.columns]
    df = df.sort_values(sort_cols_gn, kind="mergesort", na_position="last").reset_index(drop=True)
    gcols = [c for c in GAME_NO_GROUP_KEY if c in df.columns]
    df["game_no"] = df.groupby(gcols, sort=False).cumcount() + 1
    match_key = gcols  # 검증용 중복 카운트 키

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
    if len(dup_key) >= 5 and df.duplicated(subset=dup_key).any():
        print(f"[검증] 경고: 키 {dup_key} 기준 완전 중복 행이 있습니다.")

    # build_features 시계열과 동일: 연도 → 경기날짜 → game_no (동순은 경기일시 순)
    sort_cols = [c for c in SORT_OUTPUT if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_ts_for_sort"], errors="ignore")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    dh_groups = int((df.groupby(match_key, observed=True).size() > 1).sum())
    rain_null = int(df[rain_col].isna().sum())
    print("-" * 50)
    print(f"saved: {output_path}")
    print(f"rows: {len(df)}, cols: {len(df.columns)}")
    print(f"inputs: {[str(p) for p in input_paths]}")
    print(f"sort_output: {' → '.join([c for c in sort_cols if c != '_ts_for_sort'])}")
    print(f"더블헤더 추정(동일 연도·경기날짜·홈·방문·구장 키가 2행 이상인 날·대진): {dh_groups}건")
    print(f"rain_missing_after_fill: {rain_null}")
    if "구장" in df.columns:
        print("구장별 행 수 (상위 5):")
        print(df["구장"].value_counts().head(5).to_string())
    print("-" * 50)


if __name__ == "__main__":
    main()
