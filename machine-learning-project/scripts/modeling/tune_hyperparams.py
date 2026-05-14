"""
하이퍼파라미터 튜닝: Optuna + TimeSeriesSplit

- 탐색 알고리즘: TPE (Tree-structured Parzen Estimator) — 베이지안 최적화
- 교차검증:     TimeSeriesSplit(n_splits=5) — 미래 누수 방지
- 스코어:       neg_MAE (낮을수록 좋음)
- 출력:         models/best_params.json

실행:
  cd machine-learning-project
  python3 scripts/modeling/tune_hyperparams.py [--n-trials 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common.logging_config import setup_logging
from modeling.train_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TIME_KEYS,
    load_training_table,
    temporal_train_test_index,
)

N_SPLITS = 5
TEST_SIZE = 0.2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_tuning_pipeline(params: dict) -> TransformedTargetRegressor:
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
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
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


def make_objective(X_train: pd.DataFrame, y_train: pd.Series):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth":         trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_float("max_features", 0.3, 1.0),
        }
        pipe = build_tuning_pipeline(params)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        return -scores.mean()

    return objective


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna 탐색 횟수")
    args = parser.parse_args()

    root = _project_root()
    data_path = root / "data" / "processed" / "kbo_train_ready.csv"
    out_path  = root / "models" / "best_params.json"

    if not data_path.exists():
        print(f"FATAL: {data_path} 없음 — build_features.py 먼저 실행", file=sys.stderr)
        sys.exit(1)

    df = load_training_table(data_path)
    idx_train, idx_test = temporal_train_test_index(df, TEST_SIZE)
    X_train = df.loc[idx_train, NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = df.loc[idx_train, TARGET]
    X_test  = df.loc[idx_test,  NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test  = df.loc[idx_test,  TARGET]

    print("-" * 50)
    print(f"Optuna 탐색 시작 — n_trials={args.n_trials}, CV splits={N_SPLITS}")
    print(f"학습 샘플: {len(X_train)}, 테스트 샘플: {len(X_test)}")
    print("-" * 50)

    # Optuna 로그 레벨 낮춤 (WARNING만 출력)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_objective(X_train, y_train),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_cv_mae = study.best_value

    print(f"\n최적 파라미터: {best}")
    print(f"CV MAE (5-fold avg): {best_cv_mae:.2f}")

    # 최적 파라미터로 테스트 세트 검증
    pipe = build_tuning_pipeline(best)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2  = 1 - np.sum((y_test.to_numpy() - y_pred) ** 2) / np.sum(
        (y_test.to_numpy() - y_test.mean()) ** 2
    )

    print(f"테스트 MAE:  {test_mae:.2f}")
    print(f"테스트 R²:   {test_r2:.4f}")

    payload = {
        "best_params": best,
        "cv_mae_mean": float(best_cv_mae),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "n_trials": args.n_trials,
        "n_cv_splits": N_SPLITS,
        "cv_method": "TimeSeriesSplit",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {out_path}")
    print()
    print("다음 단계: train_model.py 실행 시 best_params.json 자동 반영됩니다.")
    print("-" * 50)


if __name__ == "__main__":
    main()
