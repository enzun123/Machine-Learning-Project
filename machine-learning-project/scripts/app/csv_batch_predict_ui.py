"""
Streamlit: 경기 일정 CSV → 미래(예정) 경기별 예상 관중수 예측 (단일 경기 UI와 동일 레이아웃).
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from common.congestion_levels import classify_congestion_pct
from common.stadium_capacity import venue_clip_capacity
from modeling.batch_feature_builder import (
    build_features_from_schedule,
    read_schedule_csv_bytes,
    schedule_template_path,
)
from modeling.batch_predict import predict_batch
from modeling.train_model import TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_RESULT_CORE = ("경기날짜", "홈팀", "방문팀", "구장", "관중수", "예측_관중수", "오차")


def _read_uploaded_schedule(uploaded) -> pd.DataFrame:
    uploaded.seek(0)
    data = uploaded.read()
    uploaded.seek(0)
    return read_schedule_csv_bytes(data)


def _run_predictions(
    sched: pd.DataFrame,
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
    cap_map: dict[str, int],
) -> pd.DataFrame:
    train_path = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
    tr = pd.read_csv(train_path, encoding="utf-8-sig")
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


def _format_result_table(result: pd.DataFrame) -> pd.DataFrame:
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


def _schedule_summary_line(sched: pd.DataFrame) -> str:
    n = len(sched)
    dates = pd.to_datetime(sched["경기날짜"], errors="coerce").dropna()
    if len(dates) == 0:
        return f"{n:,}경기"
    dmin = dates.min().strftime("%Y-%m-%d")
    dmax = dates.max().strftime("%Y-%m-%d")
    if dmin == dmax:
        return f"{n:,}경기 | {dmin}"
    return f"{n:,}경기 | {dmin} ~ {dmax}"


def _avg_congestion_pct(result: pd.DataFrame, cap_by: dict[str, int]) -> float:
    if "예측_관중수" not in result.columns:
        return 0.0
    pcts: list[float] = []
    for _, r in result.iterrows():
        cap = max(1, venue_clip_capacity(str(r["구장"]), str(r["홈팀"]), cap_by))
        pred = float(r["예측_관중수"])
        pcts.append(pred / cap * 100.0)
    return float(sum(pcts) / len(pcts)) if pcts else 0.0


def _plot_batch_attendance_bars(result: pd.DataFrame) -> None:
    if "예측_관중수" not in result.columns or len(result) == 0:
        return
    chart = result.copy()
    chart["_dt"] = pd.to_datetime(chart["경기날짜"], errors="coerce")
    chart = chart.sort_values("_dt", na_position="last")
    chart["경기정보"] = chart.apply(
        lambda r: (
            f"{pd.Timestamp(r['_dt']).strftime('%m/%d') if pd.notna(r['_dt']) else '?'}\n"
            f"{r['홈팀']} vs {r['방문팀']}"
        ),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(11, max(3.5, min(6.0, 0.45 * len(chart) + 2))))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")
    vals = pd.to_numeric(chart["예측_관중수"], errors="coerce").fillna(0)
    bars = ax.bar(chart["경기정보"], vals, color="#4f8cff")
    _ymax = float(vals.max()) or 1.0
    _label_dy = max(_ymax * 0.015, 200.0)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + _label_dy,
            f"{int(h):,}",
            ha="center",
            color="white",
            fontsize=9,
        )
    ax.set_title("경기별 예상 관중 수", color="white")
    ax.set_ylabel("관중 수", color="white")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#9fb3c8")
    plt.xticks(rotation=0, color="white")
    plt.yticks(color="white")
    st.pyplot(fig)
    plt.close(fig)


def process_sidebar_schedule_csv(
    uploaded,
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
    cap_by_stadium: dict[str, int],
    ml_train_ok: bool,
) -> bool:
    """사이드바 CSV 업로드 시 자동 예측. 메인에 일괄 결과를 보여줄 때 True."""
    if uploaded is None:
        for key in ("batch_attendance_result", "batch_upload_sig", "batch_ml_labels"):
            st.session_state.pop(key, None)
        return False

    sig = f"{uploaded.name}:{uploaded.size}"
    if (
        st.session_state.get("batch_upload_sig") == sig
        and "batch_attendance_result" in st.session_state
    ):
        return True

    if not ml_train_ok:
        st.sidebar.error(
            "`kbo_train_ready.csv` 없음 — `python3 scripts/features/build_features.py` 실행"
        )
        return False
    if not chosen:
        st.sidebar.warning("ML 알고리즘을 하나 이상 선택한 뒤 CSV를 올려 주세요.")
        return False

    try:
        sched = _read_uploaded_schedule(uploaded)
        with st.spinner("일정 CSV 예측 중…"):
            result = _run_predictions(
                sched,
                chosen,
                default_temp=default_temp,
                default_rain=default_rain,
                default_hum=default_hum,
                cap_map=cap_by_stadium,
            )
        st.session_state["batch_attendance_result"] = result
        st.session_state["batch_upload_sig"] = sig
        st.session_state["batch_ml_labels"] = ", ".join(chosen)
        return True
    except (KeyError, ValueError, FileNotFoundError) as e:
        st.sidebar.error(str(e))
        return False
    except Exception as e:
        st.sidebar.error(f"예측 실패: {e}")
        return False


def render_csv_batch_results_main(
    *,
    chosen: list[str],
    cap_by_stadium: dict[str, int],
    ml_train_ok: bool,
) -> None:
    """메인 영역 — 일괄 예측 결과만 표시."""
    st.markdown(
        '<div class="main-title">📈 KBO 관람 수요 예측 시스템</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-text">'
        "<b>일정 CSV</b> 업로드로 계산한 경기별 예상 관중 수입니다. "
        "다른 입력을 쓰려면 사이드바에서 CSV를 제거하세요."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not ml_train_ok:
        st.error(
            "`data/processed/kbo_train_ready.csv` 없음 — "
            "먼저 `python3 scripts/features/build_features.py` 를 실행하세요."
        )
        return

    if "batch_attendance_result" not in st.session_state:
        st.info("사이드바에서 **경기 일정 CSV**를 올려 주세요.")
        return

    result = st.session_state["batch_attendance_result"]
    if "예측_관중수" not in result.columns:
        return

    avg_att = int(round(float(result["예측_관중수"].mean())))
    n_games = len(result)
    congestion = _avg_congestion_pct(result, cap_by_stadium)
    _plan = classify_congestion_pct(congestion)

    st.markdown("## 📊 일괄 예측 요약")

    _ml_labels = st.session_state.get("batch_ml_labels", ", ".join(chosen))
    st.caption(f"예측에 **{_ml_labels}** 모델을 반영했습니다 (경기별 = 알고리즘 평균).")

    st.markdown(
        f"""
<div class="predict-box">
<div class="predict-label">평균 예상 관중 수 ({n_games:,}경기)</div>
<div class="predict-number">{avg_att:,} 명</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="disclaimer-under-predict">
<b>면책·안내.</b> 예상 관중은 <b>경기가 정상 개최된다는 전제</b>의 수요 추정입니다.
모델은 2024–25 시즌 데이터로 학습되었으며, 미래 시즌·신규 구장 일정은 오차가 클 수 있습니다.
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
<div class="card">
<div class="card-title">📋 경기 수</div>
<div class="card-value">{n_games:,} 경기</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="card">
<div class="card-title">📊 평균 혼잡도</div>
<div class="card-value">{congestion:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
<div class="card">
<div class="card-title">🛡 운영 강도 (평균)</div>
<div class="card-high">{html.escape(_plan.level)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    pred_cols = [
        c
        for c in result.columns
        if c.startswith("예측_") and c not in ("예측_관중수", "예측_평균")
    ]
    if len(pred_cols) > 1:
        with st.expander("알고리즘별 예측 비교 (경기별)", expanded=True):
            st.dataframe(
                result[["경기날짜", "홈팀", "방문팀", "구장"] + pred_cols + ["예측_관중수"]],
                hide_index=True,
                use_container_width=True,
            )

    _feat_imp_fn = st.session_state.get("batch_feat_imp_renderer")
    if chosen and _feat_imp_fn is not None:
        _feat_imp_fn(
            chosen,
            context_note=(
                f"이번 CSV **{n_games:,}경기** 일괄 예측의 평균 **{avg_att:,}명**은"
            ),
            selectbox_key="batch_ml_feat_imp_model",
        )

    st.markdown("## 📊 경기별 예상 관중 수")
    _plot_batch_attendance_bars(result)

    st.markdown("## 📋 경기별 상세")
    st.dataframe(_format_result_table(result), hide_index=True, use_container_width=True)

    if TARGET in result.columns and "오차" in result.columns and result["오차"].notna().any():
        st.metric("실제 관중 대비 평균 오차 (MAE)", f"{float(result['오차'].mean()):,.0f} 명")

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=result.to_csv(index=False, encoding="utf-8-sig"),
        file_name="kbo_attendance_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_csv_batch_predict_ui(
    *,
    chosen: list[str],
    default_temp: float,
    default_rain: float,
    default_hum: float,
    cap_by_stadium: dict[str, int],
    ml_train_ok: bool,
) -> None:
    st.markdown(
        '<motionless class="main-title">📈 KBO 관람 수요 예측 시스템</motionless>'.replace(
            "motionless", "div"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-text">'
        "사이드바에서 <b>ML 알고리즘</b>·<b>기온·강수·습도</b>를 설정한 뒤, "
        "아래에 <b>경기 일정 CSV</b>를 올리면 경기별 예상 관중 수를 한 번에 계산합니다."
        "</motionless>".replace("motionless", "div"),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if not ml_train_ok:
        st.error(
            "`data/processed/kbo_train_ready.csv` 없음 — "
            "먼저 `python3 scripts/features/build_features.py` 를 실행하세요."
        )
        return

    if not chosen:
        st.info("사이드바에서 **ML 알고리즘**을 하나 이상 선택한 뒤 CSV를 업로드하세요.")
        return

    st.markdown("## 📅 일정 CSV 업로드")

    col_up, col_dl = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "경기 일정 CSV",
            type=["csv"],
            help="필수 열: 경기날짜, 홈팀, 방문팀, 구장 · 엑셀은 CSV UTF-8로 저장",
        )
    with col_dl:
        sched_tpl = schedule_template_path(PROJECT_ROOT)
        if sched_tpl.is_file():
            st.download_button(
                "샘플 CSV",
                data=sched_tpl.read_bytes(),
                file_name=sched_tpl.name,
                mime="text/csv",
                use_container_width=True,
            )

    st.caption(
        "필수: `경기날짜`, `홈팀`, `방문팀`, `구장`  ·  "
        "선택: `기온`, `강수`, `습도`, `관중수`(비교용)"
    )

    sched: pd.DataFrame | None = None
    if uploaded is not None:
        try:
            sched = _read_uploaded_schedule(uploaded)
        except (KeyError, ValueError) as e:
            st.error(str(e))
            return

        st.markdown(
            f'<div class="match-text">{html.escape(_schedule_summary_line(sched))}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("업로드 일정 미리보기", expanded=False):
            st.dataframe(sched, hide_index=True, use_container_width=True)

    predict_clicked = st.button(
        "예상 관중수 계산",
        type="primary",
        use_container_width=True,
        disabled=sched is None,
    )

    if predict_clicked and sched is not None:
        with st.spinner("미래 경기 관중수 예측 중…"):
            try:
                result = _run_predictions(
                    sched,
                    chosen,
                    default_temp=default_temp,
                    default_rain=default_rain,
                    default_hum=default_hum,
                    cap_map=cap_by_stadium,
                )
                st.session_state["batch_attendance_result"] = result
                st.session_state["batch_ml_labels"] = ", ".join(chosen)
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"예측 실패: {e}")

    if "batch_attendance_result" not in st.session_state:
        if uploaded is None:
            st.info("일정 CSV를 올린 뒤 **예상 관중수 계산**을 누르세요.")
        return

    result = st.session_state["batch_attendance_result"]
    if "예측_관중수" not in result.columns:
        return

    avg_att = int(round(float(result["예측_관중수"].mean())))
    n_games = len(result)
    congestion = _avg_congestion_pct(result, cap_by_stadium)
    _plan = classify_congestion_pct(congestion)

    st.markdown("---")
    st.markdown("## 📊 일괄 예측 요약")

    _ml_labels = st.session_state.get("batch_ml_labels", ", ".join(chosen))
    st.caption(f"예측에 **{_ml_labels}** 모델을 반영했습니다 (경기별 = 알고리즘 평균).")

    st.markdown(
        f"""
<div class="predict-box">
<div class="predict-label">평균 예상 관중 수 ({n_games:,}경기)</div>
<div class="predict-number">{avg_att:,} 명</div>
</motionless>
""".replace("motionless", "div"),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="disclaimer-under-predict">
<b>면책·안내.</b> 예상 관중은 <b>경기가 정상 개최된다는 전제</b>의 수요 추정입니다.
모델은 2024–25 시즌 데이터로 학습되었으며, 미래 시즌·신규 구장 일정은 오차가 클 수 있습니다.
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
<div class="card">
<div class="card-title">📋 경기 수</div>
<div class="card-value">{n_games:,} 경기</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="card">
<div class="card-title">📊 평균 혼잡도</div>
<div class="card-value">{congestion:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
<div class="card">
<div class="card-title">🛡 운영 강도 (평균)</div>
<div class="card-high">{html.escape(_plan.level)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    pred_cols = [c for c in result.columns if c.startswith("예측_") and c not in ("예측_관중수", "예측_평균")]
    if len(pred_cols) > 1:
        with st.expander("알고리즘별 예측 비교 (경기별)", expanded=True):
            st.dataframe(
                result[["경기날짜", "홈팀", "방문팀", "구장"] + pred_cols + ["예측_관중수"]],
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("## 📊 경기별 예상 관중 수")
    _plot_batch_attendance_bars(result)

    st.markdown("## 📋 경기별 상세")
    st.dataframe(_format_result_table(result), hide_index=True, use_container_width=True)

    if TARGET in result.columns and "오차" in result.columns and result["오차"].notna().any():
        st.metric("실제 관중 대비 평균 오차 (MAE)", f"{float(result['오차'].mean()):,.0f} 명")

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=result.to_csv(index=False, encoding="utf-8-sig"),
        file_name="kbo_attendance_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
