"""
모델링 1단계: kbo_train_ready.csv 로 회귀 모델 학습 (가장 먼저 실행)

입력: data/processed/kbo_train_ready.csv  (build_features.py 실행 후)
출력: models/attendance_rf_pipeline.joblib, test_indices.npy, train_report.json

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    "weekday_sin",
    "weekday_cos",
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

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def load_training_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")
    df = df.dropna(subset=[TARGET])
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype(str).replace({"nan": "missing"}).fillna("missing")
    return df


def build_pipeline() -> Pipeline:
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
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", reg)])


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

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "data_path": str(data_path),
        "baseline_dummy_mean": _metrics(y_test, y_base),
        "random_forest": _metrics(y_test, y_pred),
    }

    model_path = out_dir / "attendance_rf_pipeline.joblib"
    idx_path = out_dir / "test_indices.npy"
    json_path = out_dir / "train_report.json"

    joblib.dump(pipe, model_path)
    np.save(idx_path, idx_test.to_numpy())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print("[1단계] 학습 완료 — RandomForest 회귀")
    print(f"모델: {model_path}")
    print(f"테스트 인덱스: {idx_path}")
    print(f"리포트: {json_path}")
    print(
        f"베이스라인(평균) MAE: {report['baseline_dummy_mean']['mae']:.2f} | "
        f"RMSE: {report['baseline_dummy_mean']['rmse']:.2f} | "
        f"R²: {report['baseline_dummy_mean']['r2']:.4f}"
    )
    print(
        f"RandomForest   MAE: {report['random_forest']['mae']:.2f} | "
        f"RMSE: {report['random_forest']['rmse']:.2f} | "
        f"R²: {report['random_forest']['r2']:.4f}"
    )
    print("-" * 50)


if __name__ == "__main__":
    main()
