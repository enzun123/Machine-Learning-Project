"""
모델링 2단계: train_model.py 산출물로 테스트 세트 재평가

출력: models/eval_report.json
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
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("-" * 50)
    print("[2단계] 테스트 세트 평가")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

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

    out_eval = root / "models" / "eval_report.json"
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "permutation_importance_top10": imp.head(10).to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n저장: {out_eval}")
    print("다음: python3 scripts/modeling/predict.py")
    print("-" * 50)


if __name__ == "__main__":
    main()
