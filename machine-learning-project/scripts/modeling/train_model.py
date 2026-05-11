"""
모델링 1단계: kbo_train_ready.csv 로 회귀 모델 학습

- 검증: 연도·월·주차_ISO 기준 시간 순 홀드아웃 (마지막 구간을 테스트)
- 베이스라인: 전역 평균(Dummy), 구장별 평균 관중
- 모델: RandomForest 회귀 파이프라인

입력: data/processed/kbo_train_ready.csv
출력: models/attendance_rf_pipeline.joblib, test_indices.npy, train_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

TARGET = "관중수"

NUMERIC_FEATURES = [
    "연도",
    "월",
    "주차_ISO",
    "stadium_capacity",
    "is_capacity_missing",
    "is_rain",
    "is_hot",
    "is_weekend",
    "is_friday",            # [수정5]
    "is_saturday",          # [수정5]
    "is_sunday",            # [수정5]
    "weekday_sin",
    "weekday_cos",
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
    "home_prior_mean_att",
    "visitor_prior_mean_att",
    "home_last5_mean_att",
    "visitor_last5_mean_att",
    "home_draw_pct_in_league",
    "visitor_away_draw_pct_in_league",
    "home_visitor_prior_draw_diff",
    "matchup_prior_mean_att",    # [수정6]
    "season_progress",           # [수정4]
]

CATEGORICAL_FEATURES = [
    "홈팀",
    "방문팀",
    "구장",
    "rain_bucket",
    "temp_bucket",
    "humidity_bucket",
    "wind_bucket",
    "stadium_x_rain",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TIME_KEYS = ["연도", "월", "주차_ISO"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}


def load_training_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in FEATURE_COLUMNS + [TARGET] + TIME_KEYS if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")
    df = df.dropna(subset=[TARGET])
    for c in TIME_KEYS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype(str).replace({"nan": "missing"}).fillna("missing")
    return df


def temporal_train_test_index(df: pd.DataFrame, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """연도 → 월 → 주차_ISO 순으로 정렬한 뒤, 시간상 뒤쪽 비율을 테스트로 사용."""
    n = len(df)
    n_test = min(max(1, int(round(n * test_size))), n - 1)
    order = np.lexsort(
        (
            df["주차_ISO"].to_numpy(),
            df["월"].to_numpy(),
            df["연도"].to_numpy(),
        )
    )
    sorted_idx = df.index.to_numpy()[order]
    test_idx = sorted_idx[-n_test:]
    train_idx = sorted_idx[:-n_test]
    return train_idx, test_idx


def stadium_mean_baseline_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """학습 구간 구장별 평균 관중; 미관측 구장은 학습 전역 평균."""
    gmean = (
        pd.DataFrame({"구장": X_train["구장"].values, "y": y_train.values})
        .groupby("구장", observed=True)["y"]
        .mean()
    )
    overall = float(y_train.mean())
    mapped = X_test["구장"].map(gmean)
    return mapped.fillna(overall).to_numpy(dtype=float)


def _load_best_params(root: Path) -> dict:
    """models/best_params.json 이 있으면 최적 파라미터 반환, 없으면 기본값."""
    p = root / "models" / "best_params.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("best_params", {})
        print(f"[best_params.json 적용] {params}")
        return params
    return {}


def build_pipeline(root: Path | None = None) -> TransformedTargetRegressor:
    """
    RandomForest + log1p 타깃 변환.
    models/best_params.json 가 있으면 튜닝된 파라미터 자동 적용.
    없으면 기본값(n_estimators=300, max_depth=15, min_samples_leaf=4) 사용.
    """
    defaults = {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 4, "max_features": 1.0}
    if root is not None:
        tuned = _load_best_params(root)
        defaults.update(tuned)

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    reg = RandomForestRegressor(
        n_estimators=int(defaults["n_estimators"]),
        max_depth=int(defaults["max_depth"]) if defaults["max_depth"] is not None else None,
        min_samples_leaf=int(defaults["min_samples_leaf"]),
        max_features=float(defaults["max_features"]),
        criterion="absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    inner = Pipeline([("preprocess", pre), ("model", reg)])
    return TransformedTargetRegressor(
        regressor=inner,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def _split_time_bounds(df: pd.DataFrame, idx: np.ndarray) -> dict:
    sub = df.loc[idx, TIME_KEYS]
    return {
        "연도_min": int(sub["연도"].min()),
        "연도_max": int(sub["연도"].max()),
        "월_min": int(sub["월"].min()),
        "월_max": int(sub["월"].max()),
        "주차_ISO_min": int(sub["주차_ISO"].min()),
        "주차_ISO_max": int(sub["주차_ISO"].max()),
        "rows": int(len(idx)),
    }


def main() -> None:
    root = _project_root()
    data_path = root / "data" / "processed" / "kbo_train_ready.csv"
    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(
            f"FATAL: {data_path} 가 없습니다.\n"
            "먼저 실행: python3 scripts/features/build_features.py",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_training_table(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    idx_train, idx_test = temporal_train_test_index(df, TEST_SIZE)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_dummy = baseline.predict(X_test)

    y_stadium = stadium_mean_baseline_predict(X_train, y_train, X_test)

    pipe = build_pipeline(root)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": TARGET,
        "split": "temporal_연도_월_주차_ISO",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "data_path": str(data_path),
        "train_time_bounds": _split_time_bounds(df, idx_train),
        "test_time_bounds": _split_time_bounds(df, idx_test),
        "baseline_dummy_mean": _metrics(y_test, y_dummy),
        "baseline_stadium_mean": _metrics(y_test, y_stadium),
        "model": _metrics(y_test, y_pred),
        "model_type": "RandomForest",
        "log_target_transform": True,
    }

    model_path = out_dir / "attendance_rf_pipeline.joblib"
    idx_path = out_dir / "test_indices.npy"
    json_path = out_dir / "train_report.json"

    joblib.dump(pipe, model_path)
    np.save(idx_path, np.asarray(idx_test))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    mtype = report["model_type"]
    print(f"[1단계] 학습 완료 — {mtype} + log1p 타깃 변환 (시간 순 테스트)")
    print(f"테스트 기간(대략): 연도 {report['test_time_bounds']['연도_min']}–{report['test_time_bounds']['연도_max']}")
    print(f"모델: {model_path}")
    print(f"테스트 인덱스: {idx_path}")
    print(f"리포트: {json_path}")
    print(
        f"베이스라인(전역 평균) MAE: {report['baseline_dummy_mean']['mae']:.2f} | "
        f"RMSE: {report['baseline_dummy_mean']['rmse']:.2f} | "
        f"MSE: {report['baseline_dummy_mean']['mse']:.2f}"
    )
    print(
        f"베이스라인(구장 평균) MAE: {report['baseline_stadium_mean']['mae']:.2f} | "
        f"RMSE: {report['baseline_stadium_mean']['rmse']:.2f} | "
        f"R²: {report['baseline_stadium_mean']['r2']:.4f}"
    )
    print(
        f"{mtype:18s} MAE: {report['model']['mae']:.2f} | "
        f"RMSE: {report['model']['rmse']:.2f} | "
        f"R²: {report['model']['r2']:.4f}"
    )
    print("-" * 50)


if __name__ == "__main__":
    main()
