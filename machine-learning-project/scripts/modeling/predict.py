"""
모델링 3단계: 저장된 파이프라인으로 관중수 예측

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_model import FEATURE_COLUMNS, TARGET, load_training_table


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_model(path: Path | None = None):
    root = _project_root()
    p = path or (root / "models" / "attendance_rf_pipeline.joblib")
    if not p.exists():
        raise FileNotFoundError(f"모델 없음: {p} — train_model.py 먼저 실행")
    return joblib.load(p)


def predict_df(pipe, X: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise KeyError(f"입력 컬럼 누락: {missing}")
    X = X[FEATURE_COLUMNS].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).fillna("missing")
    return np.asarray(pipe.predict(X))


def main() -> None:
    parser = argparse.ArgumentParser(description="관중수 회귀 예측")
    parser.add_argument("--model", type=str, default=None, help=".joblib 경로")
    args = parser.parse_args()

    root = _project_root()
    model_path = Path(args.model) if args.model else root / "models" / "attendance_rf_pipeline.joblib"
    pipe = load_model(model_path)

    csv_path = root / "data" / "processed" / "kbo_train_ready.csv"
    if not csv_path.exists():
        print(f"CSV 없음: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = load_training_table(csv_path)
    one = df[FEATURE_COLUMNS].iloc[[0]]
    pred = predict_df(pipe, one)[0]
    print(f"샘플 1행 예측 관중수: {pred:,.0f} 명 (실제: {df[TARGET].iloc[0]:,.0f})")


if __name__ == "__main__":
    main()
