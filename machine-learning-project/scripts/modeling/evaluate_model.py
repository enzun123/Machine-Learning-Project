"""
모델링 2단계: train_model.py 산출물로 테스트 세트 재평가

출력: models/eval_report.json (잔차 요약, 구장별 MAE 포함)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_model import FEATURE_COLUMNS, TARGET, load_training_table

RANDOM_STATE = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _project_root()
    data_path = root / "data" / "processed" / "kbo_train_ready.csv"
    model_path = root / "models" / "attendance_rf_pipeline.joblib"
    idx_path = root / "models" / "test_indices.npy"
    train_report_path = root / "models" / "train_report.json"

    if not model_path.exists():
        print(f"FATAL: 모델 없음 — 먼저 실행: python3 scripts/modeling/train_model.py\n{model_path}", file=sys.stderr)
        sys.exit(1)
    if not idx_path.exists():
        print(f"FATAL: test_indices.npy 없음 — train_model.py 재실행\n{idx_path}", file=sys.stderr)
        sys.exit(1)

    df = load_training_table(data_path)
    test_idx = np.load(idx_path)
    d_test = df.loc[test_idx]
    X_test = d_test[FEATURE_COLUMNS]
    y_test = d_test[TARGET]

    pipe = joblib.load(model_path)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    residual = y_test.to_numpy(dtype=float) - y_pred

    mae_by_stadium = (
        pd.DataFrame({"구장": d_test["구장"].values, "ae": np.abs(residual)})
        .groupby("구장", observed=True)["ae"]
        .mean()
        .sort_values(ascending=False)
    )

    split_info = {}
    if train_report_path.exists():
        with open(train_report_path, encoding="utf-8") as f:
            tr = json.load(f)
        split_info = {
            "split": tr.get("split"),
            "test_time_bounds": tr.get("test_time_bounds"),
        }

    print("-" * 50)
    print("[2단계] 테스트 세트 평가")
    if split_info.get("split"):
        print(f"분할 방식: {split_info['split']}")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"잔차 |평균|: {np.mean(np.abs(residual)):.2f} | 최대 절대오차: {np.max(np.abs(residual)):.2f}")

    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nPermutation importance (상위 10개):")
    print(imp.head(10).to_string())

    print("\n구장별 평균 절대오차 (상위 5)")
    print(mae_by_stadium.head(5).to_string())

    out_eval = root / "models" / "eval_report.json"
    payload = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
        "permutation_importance_top10": imp.head(10).to_dict(),
        "mae_by_stadium_top15": mae_by_stadium.head(15).to_dict(),
        **split_info,
    }
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_eval}")
    print("-" * 50)


if __name__ == "__main__":
    main()
