"""
RF / LightGBM / XGBoost 동일 시간순 분할·피처로 학습·비교·저장.

출력:
  models/attendance_rf_pipeline.joblib   (RF, train_model과 동일 구조)
  models/attendance_lgbm_pipeline.joblib
  models/attendance_xgb_pipeline.joblib
  models/test_indices.npy
  reports/modeling/model_benchmark.json

실행:
  cd machine-learning-project
  PYTHONPATH=scripts python3 scripts/modeling/benchmark_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from common.logging_config import setup_logging
from modeling.train_model import (
    FEATURE_COLUMNS,
    TARGET,
    TEST_SIZE,
    _metrics,
    _project_root,
    _split_time_bounds,
    build_pipeline,
    load_training_table,
    stadium_mean_baseline_predict,
    temporal_train_test_index,
)
from modeling.tune_hyperparams import (
    HAS_LGBM,
    HAS_XGB,
    build_lgbm_pipeline,
    build_xgb_pipeline,
)

# streamlit_app 과 동일 파일명
MODEL_FILES = {
    "RandomForest": "attendance_rf_pipeline.joblib",
    "LightGBM": "attendance_lgbm_pipeline.joblib",
    "XGBoost": "attendance_xgb_pipeline.joblib",
}


def _load_best_params(root: Path, json_name: str) -> dict | None:
    p = root / "models" / json_name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params")


def _lgbm_pipeline(root: Path) -> TransformedTargetRegressor:
    tuned = _load_best_params(root, "best_lgbm_params.json")
    if tuned:
        print(f"[best_lgbm_params.json 적용] {tuned}")
        return build_lgbm_pipeline(tuned)
    print("[LightGBM] 기본 하이퍼파라미터 (튜닝 없음 — tune_hyperparams.py --model lgbm)")
    return build_lgbm_pipeline(
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    )


def _xgb_pipeline(root: Path) -> TransformedTargetRegressor:
    tuned = _load_best_params(root, "best_xgb_params.json")
    if tuned:
        print(f"[best_xgb_params.json 적용] {tuned}")
        return build_xgb_pipeline(tuned)
    print("[XGBoost] 기본 하이퍼파라미터 (튜닝 없음 — tune_hyperparams.py --model xgb)")
    return build_xgb_pipeline(
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    )


def main() -> None:
    setup_logging()
    root = _project_root()
    data_path = root / "data" / "processed" / "kbo_train_ready.csv"
    models_dir = root / "models"
    reports_dir = root / "reports" / "modeling"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"FATAL: {data_path} 없음 — build_features.py 먼저 실행", file=sys.stderr)
        sys.exit(1)

    df = load_training_table(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    idx_train, idx_test = temporal_train_test_index(df, TEST_SIZE)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_stadium = stadium_mean_baseline_predict(X_train, y_train, X_test)

    candidates: list[tuple[str, TransformedTargetRegressor]] = [
        ("RandomForest", build_pipeline(root)),
    ]
    if HAS_LGBM:
        candidates.append(("LightGBM", _lgbm_pipeline(root)))
    else:
        print("[skip] LightGBM 미설치")
    if HAS_XGB:
        candidates.append(("XGBoost", _xgb_pipeline(root)))
    else:
        print("[skip] XGBoost 미설치 (pip install xgboost)")

    results: dict = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split": "temporal_연도_월_주차_ISO",
        "baseline_dummy_mean": _metrics(y_test, dummy.predict(X_test)),
        "baseline_stadium_mean": _metrics(y_test, y_stadium),
        "models": {},
        "saved_paths": {},
    }

    best_name = ""
    best_mae = float("inf")

    for name, pipe in candidates:
        print(f"학습·저장: {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        m = _metrics(y_test, y_pred)
        results["models"][name] = m

        out_name = MODEL_FILES.get(name)
        if out_name:
            out_path = models_dir / out_name
            joblib.dump(pipe, out_path)
            results["saved_paths"][name] = str(out_path)
            print(f"  → {out_path}")

        if m["mae"] < best_mae:
            best_mae = m["mae"]
            best_name = name

    np.save(models_dir / "test_indices.npy", np.asarray(idx_test))
    results["best_by_mae"] = best_name
    results["test_time_bounds"] = _split_time_bounds(df, idx_test)

    out_path = reports_dir / "model_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("-" * 62)
    print(f"{'모델':<16} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 62)
    for name, m in sorted(results["models"].items(), key=lambda x: x[1]["mae"]):
        mark = "  ★" if name == best_name else ""
        print(f"{name:<16} {m['mae']:8.1f} {m['rmse']:8.1f} {m['r2']:8.4f}{mark}")
    print("-" * 62)
    print(f"베스트(MAE): {best_name}")
    print(f"리포트: {out_path}")


if __name__ == "__main__":
    main()
