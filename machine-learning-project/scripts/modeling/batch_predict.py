"""
CSV 피처 행 일괄 관중 예측 (Streamlit·CLI 공용).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling.predict import predict_df
from modeling.train_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET,
    TIME_KEYS,
)

MODEL_FILES: dict[str, str] = {
    "RandomForest": "attendance_rf_pipeline.joblib",
    "LightGBM": "attendance_lgbm_pipeline.joblib",
    "XGBoost": "attendance_xgb_pipeline.joblib",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in df.columns]


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """업로드 CSV → 모델 입력 행 (관중수 없어도 됨)."""
    missing = validate_feature_columns(df)
    if missing:
        raise KeyError(f"필수 피처 컬럼 누락: {missing}")

    out = df.copy()
    for c in TIME_KEYS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    for c in CATEGORICAL_FEATURES:
        if c in out.columns:
            out[c] = out[c].astype(str).replace({"nan": "missing"}).fillna("missing")
    return out


def _clip_to_capacity(preds: np.ndarray, capacities: pd.Series | None) -> np.ndarray:
    if capacities is None:
        return preds
    cap = pd.to_numeric(capacities, errors="coerce")
    if cap.isna().all():
        return preds
    cap_arr = cap.fillna(np.inf).to_numpy(dtype=float)
    return np.minimum(preds, cap_arr)


def load_pipeline(model_label: str, root: Path | None = None) -> object:
    root = root or _project_root()
    fname = MODEL_FILES.get(model_label)
    if not fname:
        raise ValueError(f"알 수 없는 모델: {model_label}")
    path = root / "models" / fname
    if not path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {path}")
    return joblib.load(path)


def predict_batch(
    df: pd.DataFrame,
    model_labels: list[str],
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """
    피처가 포함된 DataFrame에 대해 경기별 예측 열을 붙인다.

    반환 열 예: 예측_RandomForest, 예측_LightGBM, 예측_평균
    """
    if not model_labels:
        raise ValueError("model_labels가 비어 있습니다.")

    prepared = prepare_feature_frame(df)
    X = prepared[FEATURE_COLUMNS]
    cap = prepared["stadium_capacity"] if "stadium_capacity" in prepared.columns else None

    out = df.copy()
    pred_cols: list[str] = []

    for label in model_labels:
        pipe = load_pipeline(label, root=root)
        raw = predict_df(pipe, X)
        clipped = _clip_to_capacity(raw, cap)
        col = f"예측_{label}"
        out[col] = np.round(clipped).astype(int)
        pred_cols.append(col)

    if len(pred_cols) > 1:
        out["예측_평균"] = out[pred_cols].mean(axis=1).round().astype(int)
        out["예측_관중수"] = out["예측_평균"]
    elif len(pred_cols) == 1:
        out["예측_관중수"] = out[pred_cols[0]]

    if TARGET in out.columns and pred_cols:
        ref_col = "예측_평균" if "예측_평균" in out.columns else pred_cols[0]
        actual = pd.to_numeric(out[TARGET], errors="coerce")
        if actual.notna().any():
            out["오차"] = (out[ref_col] - actual).abs().round().astype("Int64")

    return out


def feature_template_path(root: Path | None = None) -> Path:
    return (root or _project_root()) / "data" / "external" / "batch_predict_feature_template.csv"
