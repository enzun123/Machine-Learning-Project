"""
하이퍼파라미터 튜닝: Optuna + TimeSeriesSplit

- 탐색 알고리즘: TPE — 베이지안 최적화
- 교차검증:     TimeSeriesSplit(n_splits=5) — 미래 누수 방지
- 스코어:       neg_MAE (낮을수록 좋음)
- 출력:
    models/best_params.json       (RandomForest, train_model.py 가 읽음)
    models/best_lgbm_params.json  (benchmark_models.py 가 읽음)
    models/best_xgb_params.json

실행:
  cd machine-learning-project
  PYTHONPATH=scripts python3 scripts/modeling/tune_hyperparams.py
  PYTHONPATH=scripts python3 scripts/modeling/tune_hyperparams.py --model lgbm --n-trials 50
  PYTHONPATH=scripts python3 scripts/modeling/tune_hyperparams.py --model all --n-trials 50
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
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    load_training_table,
    temporal_train_test_index,
)

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

N_SPLITS = 5
TEST_SIZE = 0.2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def _wrap(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", _preprocessor()), ("model", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_tuning_pipeline(params: dict) -> TransformedTargetRegressor:
    reg = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        criterion="absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return _wrap(reg)


def build_lgbm_pipeline(params: dict) -> TransformedTargetRegressor:
    return _wrap(
        LGBMRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    )


def build_xgb_pipeline(params: dict) -> TransformedTargetRegressor:
    return _wrap(
        XGBRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    )


def make_rf_objective(X_train: pd.DataFrame, y_train: pd.Series):
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


def make_lgbm_objective(X_train: pd.DataFrame, y_train: pd.Series):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe = build_lgbm_pipeline(params)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        return -float(scores.mean())

    return objective


def make_xgb_objective(X_train: pd.DataFrame, y_train: pd.Series):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe = build_xgb_pipeline(params)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        return -float(scores.mean())

    return objective


def _run_study(
    name: str,
    objective,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    print(f"\n[{name}] Optuna n_trials={n_trials}")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  CV MAE: {study.best_value:.2f}")
    print(f"  best_params: {study.best_params}")
    return study


def _eval_test(
    pipe: TransformedTargetRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = 1 - np.sum((y_test.to_numpy() - y_pred) ** 2) / np.sum(
        (y_test.to_numpy() - y_test.mean()) ** 2
    )
    return float(test_mae), float(test_r2)


def _save_tune_result(
    out_path: Path,
    best_params: dict,
    cv_mae: float,
    test_mae: float,
    test_r2: float,
    n_trials: int,
) -> None:
    payload = {
        "best_params": best_params,
        "cv_mae_mean": float(cv_mae),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "n_trials": n_trials,
        "n_cv_splits": N_SPLITS,
        "cv_method": "TimeSeriesSplit",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}  (테스트 MAE {test_mae:.1f}, R² {test_r2:.4f})")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna 탐색 횟수 (모델당)")
    parser.add_argument(
        "--model",
        choices=("rf", "lgbm", "xgb", "all"),
        default="all",
        help="튜닝 대상 (기본: all)",
    )
    args = parser.parse_args()

    root = _project_root()
    models_dir = root / "models"
    data_path = root / "data" / "processed" / "kbo_train_ready.csv"

    if not data_path.exists():
        print(f"FATAL: {data_path} 없음 — build_features.py 먼저 실행", file=sys.stderr)
        sys.exit(1)

    df = load_training_table(data_path)
    idx_train, idx_test = temporal_train_test_index(df, TEST_SIZE)
    X_train = df.loc[idx_train, FEATURE_COLUMNS]
    y_train = df.loc[idx_train, TARGET]
    X_test = df.loc[idx_test, FEATURE_COLUMNS]
    y_test = df.loc[idx_test, TARGET]

    print("-" * 50)
    print(f"튜닝 대상: {args.model}, n_trials={args.n_trials}, CV={N_SPLITS}-fold")
    print(f"학습 {len(X_train)} / 테스트 {len(X_test)}")
    print("-" * 50)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.model in ("rf", "all"):
        study = _run_study("RandomForest", make_rf_objective(X_train, y_train), args.n_trials, RANDOM_STATE)
        pipe = build_tuning_pipeline(study.best_params)
        test_mae, test_r2 = _eval_test(pipe, X_train, y_train, X_test, y_test)
        _save_tune_result(
            models_dir / "best_params.json",
            study.best_params,
            study.best_value,
            test_mae,
            test_r2,
            args.n_trials,
        )

    if args.model in ("lgbm", "all"):
        if not HAS_LGBM:
            print("[skip] LightGBM 미설치")
        else:
            study = _run_study(
                "LightGBM",
                make_lgbm_objective(X_train, y_train),
                args.n_trials,
                RANDOM_STATE,
            )
            pipe = build_lgbm_pipeline(study.best_params)
            test_mae, test_r2 = _eval_test(pipe, X_train, y_train, X_test, y_test)
            _save_tune_result(
                models_dir / "best_lgbm_params.json",
                study.best_params,
                study.best_value,
                test_mae,
                test_r2,
                args.n_trials,
            )

    if args.model in ("xgb", "all"):
        if not HAS_XGB:
            print("[skip] XGBoost 미설치 (pip install xgboost)")
        else:
            study = _run_study(
                "XGBoost",
                make_xgb_objective(X_train, y_train),
                args.n_trials,
                RANDOM_STATE + 1,
            )
            pipe = build_xgb_pipeline(study.best_params)
            test_mae, test_r2 = _eval_test(pipe, X_train, y_train, X_test, y_test)
            _save_tune_result(
                models_dir / "best_xgb_params.json",
                study.best_params,
                study.best_value,
                test_mae,
                test_r2,
                args.n_trials,
            )

    print()
    print("다음: PYTHONPATH=scripts python3 scripts/modeling/benchmark_models.py")
    print("      (튜닝 파라미터로 LGBM/XGB joblib 재생성)")
    print("-" * 50)


if __name__ == "__main__":
    main()
