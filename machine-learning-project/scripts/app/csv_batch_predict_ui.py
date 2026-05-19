"""
Streamlit: 경기 일정 CSV → 미래(예정) 경기별 예상 관중수 예측
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modeling.batch_feature_builder import (
    build_features_from_schedule,
    load_stadium_capacity_map,
    normalize_schedule_csv,
    schedule_template_path,
)
from modeling.batch_predict import (
    feature_template_path,
    predict_batch,
    validate_feature_columns,
)
from modeling.train_model import TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 결과 표에 먼저 보여 줄 열 (관중수 예측이 핵심)
_RESULT_CORE = ("경기날짜", "홈팀", "방문팀", "구장", "관중수", "예측_관중수", "오차")


def _model_available(filename: str) -> bool:
    return (PROJECT_ROOT / "models" / filename).is_file()


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded, encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="cp949")


def _render_model_choices() -> list[str]:
    chosen: list[str] = []
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.checkbox(
            "RandomForest",
            value=_model_available("attendance_rf_pipeline.joblib"),
            disabled=not _model_available("attendance_rf_pipeline.joblib"),
        ):
            chosen.append("RandomForest")
    with c2:
        if st.checkbox(
            "LightGBM",
            value=False,
            disabled=not _model_available("attendance_lgbm_pipeline.joblib"),
        ):
            chosen.append("LightGBM")
    with c3:
        if st.checkbox(
            "XGBoost",
            value=False,
            disabled=not _model_available("attendance_xgb_pipeline.joblib"),
        ):
            chosen.append("XGBoost")
    return chosen


def _format_result_table(result: pd.DataFrame) -> pd.DataFrame:
    """관중수 예측 중심 표 (숫자는 천 단위 콤마)."""
    cols = [c for c in _RESULT_CORE if c in result.columns]
    extra = [c for c in result.columns if c.startswith("예측_") and c not in cols]
    cols = cols + [c for c in extra if c not in cols]
    out = result[cols].copy() if cols else result.copy()
    if "예측_관중수" in out.columns:
        out = out.rename(columns={"예측_관중수": "예상 관중수(명)"})
    if "관중수" in out.columns:
        out = out.rename(columns={"관중수": "실제 관중수(명)"})
    for c in out.columns:
        if "관중수" in str(c) and c != "오차":
            out[c] = pd.to_numeric(out[c], errors="coerce").apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else ""
            )
    return out


def _run_predictions(
    upload_mode: str,
    raw: pd.DataFrame,
    sched: pd.DataFrame | None,
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
) -> pd.DataFrame:
    train_path = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"

    if upload_mode.startswith("일정"):
        if sched is None:
            sched = normalize_schedule_csv(raw)
        tr = pd.read_csv(train_path, encoding="utf-8-sig")
        cap_map = load_stadium_capacity_map(PROJECT_ROOT)
        feat = build_features_from_schedule(
            sched,
            tr,
            cap_map,
            default_temp=default_temp,
            default_rain=default_rain,
            default_hum=default_hum,
        )
        pred = predict_batch(feat, chosen, root=PROJECT_ROOT)
        result = sched.copy()
        for col in pred.columns:
            if col.startswith("예측_") or col == "오차":
                result[col] = pred[col].values
        return result

    return predict_batch(raw, chosen, root=PROJECT_ROOT)


def render_csv_batch_predict_ui() -> None:
    st.header("미래 경기 예상 관중수")
    st.markdown(
        """
**예정된 경기 일정 CSV**만 있으면 됩니다. 각 경기마다 **아직 모르는 관중수를 예측**합니다.

| 필요한 것 | 설명 |
|-----------|------|
| `경기날짜`, `홈팀`, `방문팀`, `구장` | 2026 시즌 일정처럼 **앞으로 치를 경기** 목록 |
| `관중수` | **없어도 됨** (미래 경기). 있으면 예측과 비교용 |
| `기온`, `강수`, `습도` | 없으면 아래 기본값 사용 |

42개 피처 CSV는 만들 필요 없습니다. 일정만 올리면 앱이 피처를 만들고 RF/LightGBM/XGB로 **예상 관중수(명)** 를 냅니다.
        """
    )

    sched_tpl = schedule_template_path(PROJECT_ROOT)
    if sched_tpl.is_file():
        st.download_button(
            "샘플 일정 CSV 받기",
            data=sched_tpl.read_bytes(),
            file_name=sched_tpl.name,
            mime="text/csv",
        )

    st.caption(
        "필수: `경기날짜`, `홈팀`, `방문팀`, `구장`  ·  "
        "선택: `기온`, `강수`, `습도`  ·  `관중수`는 미래 예측 시 비워 두세요"
    )

    train_path = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
    if not train_path.is_file():
        st.error("`kbo_train_ready.csv` 없음 — 먼저 `python3 scripts/features/build_features.py` 실행")
        return

    st.subheader("1. 모델 선택")
    chosen = _render_model_choices()
    if not chosen:
        st.warning("예측 모델을 하나 이상 선택하세요.")
        return

    with st.expander("날씨 기본값 (CSV에 기온·강수·습도가 없을 때)", expanded=False):
        default_temp = st.number_input("기온 (℃)", value=18.0, step=0.5)
        default_rain = st.number_input("강수 (mm)", value=0.0, min_value=0.0, step=0.1)
        default_hum = st.number_input("습도 (%)", value=55.0, min_value=0.0, max_value=100.0, step=1.0)

    st.subheader("2. 일정 CSV 업로드")
    uploaded = st.file_uploader(
        "CSV 파일",
        type=["csv"],
        help="엑셀에서 저장한 UTF-8 CSV. 탭 구분이면 CSV로 다시 저장하세요.",
    )

    with st.expander("고급: 피처 42개가 이미 들어 있는 CSV (전문가용)", expanded=False):
        st.caption("`kbo_train_ready`와 동일한 컬럼이 있으면 피처 생성 없이 바로 관중수만 예측합니다.")
        feat_tpl = feature_template_path(PROJECT_ROOT)
        if feat_tpl.is_file():
            st.download_button(
                "피처 CSV 샘플",
                data=feat_tpl.read_bytes(),
                file_name=feat_tpl.name,
                mime="text/csv",
                key="dl_feat_tpl",
            )
        feat_upload = st.file_uploader("피처 CSV", type=["csv"], key="feat_csv")
        if feat_upload is not None:
            raw_feat = _read_uploaded_csv(feat_upload)
            missing = validate_feature_columns(raw_feat)
            if missing:
                st.error(f"컬럼 부족: {missing[:5]}…")
            elif st.button("관중수 예측 (피처 CSV)", type="primary", key="predict_feat"):
                with st.spinner("관중수 예측 중…"):
                    result = _run_predictions("피처", raw_feat, None, chosen, default_temp=18, default_rain=0, default_hum=55)
                st.session_state["batch_attendance_result"] = result

    if uploaded is None:
        st.info("미래·예정 경기 일정 CSV를 올린 뒤 **예상 관중수 계산**을 누르세요.")
        return

    raw = _read_uploaded_csv(uploaded)
    try:
        sched = normalize_schedule_csv(raw)
    except KeyError as e:
        st.error(str(e))
        return

    st.success(f"{len(sched):,}개 경기 인식됨")

    predict_clicked = st.button("예상 관중수 계산", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("미래 경기 관중수 예측 중…"):
            try:
                result = _run_predictions(
                    "일정",
                    raw,
                    sched,
                    chosen,
                    default_temp=float(default_temp),
                    default_rain=float(default_rain),
                    default_hum=float(default_hum),
                )
                st.session_state["batch_attendance_result"] = result
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"예측 실패: {e}")

    if "batch_attendance_result" not in st.session_state:
        return

    result = st.session_state["batch_attendance_result"]

    st.subheader("3. 결과 — 경기별 예상 관중수 (미래)")
    display = _format_result_table(result)
    st.dataframe(display, width="stretch", hide_index=True)

    if "예측_관중수" in result.columns:
        total_games = len(result)
        avg_att = int(result["예측_관중수"].mean())
        c1, c2 = st.columns(2)
        c1.metric("경기 수", f"{total_games:,} 경기")
        c2.metric("예상 관중 평균", f"{avg_att:,} 명")

    if TARGET in result.columns and "오차" in result.columns and result["오차"].notna().any():
        st.metric("실제 관중 대비 평균 오차 (MAE)", f"{float(result['오차'].mean()):,.0f} 명")

    with st.expander("알고리즘별 예측값 (상세)", expanded=False):
        detail_cols = [c for c in result.columns if c.startswith("예측_")]
        st.dataframe(result[detail_cols], width="stretch", hide_index=True)

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=result.to_csv(index=False, encoding="utf-8-sig"),
        file_name="kbo_attendance_predictions.csv",
        mime="text/csv",
    )
