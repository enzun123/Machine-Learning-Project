"""
Streamlit: 경기 일정 CSV → 미래(예정) 경기별 예상 관중수 예측 (단일 경기 UI와 동일 레이아웃).
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common.attendance_parse import attendance_sources_fingerprint, parse_attendance_column
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

try:
    _KBO_BATCH_ATT_TTL = int(os.environ.get("KBO_BATCH_ATTENDANCE_TTL_SEC", "300"))
except ValueError:
    _KBO_BATCH_ATT_TTL = 300

_RESULT_CORE = ("경기날짜", "홈팀", "방문팀", "구장", "관중수", "예측_관중수", "오차")

_ALGO_PRED_COLS: tuple[str, ...] = (
    "예측_RandomForest",
    "예측_LightGBM",
    "예측_XGBoost",
)
_ALGO_PRED_DISPLAY: dict[str, str] = {
    "예측_RandomForest": "RF 예측(명)",
    "예측_LightGBM": "LGBM 예측(명)",
    "예측_XGBoost": "XGB 예측(명)",
}

_ACTUAL_BAR_COLOR = "#4f8cff"
_PRED_BAR_COLOR_SINGLE = "#f59e0b"


def _model_bar_color(label: str) -> str:
    return {
        "RandomForest": "#a78bfa",
        "LightGBM": "#22c55e",
        "XGBoost": "#f97316",
    }.get(label, "#f59e0b")


def _slot_bar_layout(n_bars: int, *, span: float = 0.9, width_ratio: float = 0.72) -> tuple[list[float], float]:
    if n_bars <= 0:
        return [], 0.0
    bar_w = (span / n_bars) * width_ratio
    if n_bars == 1:
        return [0.0], bar_w * 1.05
    step = span / n_bars
    offs = [-span / 2 + step / 2 + i * step for i in range(n_bars)]
    return offs, bar_w


def _draw_bars_at_slot(
    ax,
    xi: float,
    bars: list[tuple[str, int, str]],
    *,
    legend_seen: set[str],
) -> tuple[float, list[float]]:
    if not bars:
        return 0.0, []
    offs, bw = _slot_bar_layout(len(bars))
    ymax = 0.0
    for off, (label, val, color) in zip(offs, bars):
        leg = label if label not in legend_seen else "_nolegend_"
        if leg != "_nolegend_":
            legend_seen.add(label)
        ax.bar(xi + off, val, bw, label=leg, color=color)
        ymax = max(ymax, float(val))
    return ymax, offs


def _algo_preds_from_row(r: pd.Series) -> dict[str, int]:
    preds: dict[str, int] = {}
    for col in _ALGO_PRED_COLS:
        if col not in r.index:
            continue
        v = pd.to_numeric(r.get(col), errors="coerce")
        if pd.notna(v):
            preds[col.replace("예측_", "", 1)] = int(round(float(v)))
    return preds


def _ordered_batch_algo_preds(preds: dict[str, int]) -> list[tuple[str, int]]:
    order = [col.replace("예측_", "", 1) for col in _ALGO_PRED_COLS]
    return [(label, preds[label]) for label in order if label in preds]


def _ml_models_session_key(chosen: list[str]) -> str:
    return ",".join(sorted(chosen))


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
    if TARGET in sched.columns:
        result[TARGET] = _parse_attendance_series(sched[TARGET])
    for col in pred.columns:
        if col.startswith("예측_") or col == "오차":
            result[col] = pred[col].values
    if TARGET in result.columns and "예측_관중수" in result.columns:
        ref = "예측_평균" if "예측_평균" in result.columns else "예측_관중수"
        mask = result[TARGET].notna()
        if mask.any():
            result.loc[mask, "오차"] = (
                pd.to_numeric(result.loc[mask, ref], errors="coerce")
                - result.loc[mask, TARGET]
            ).abs().round().astype("Int64")
    return result


def _format_result_table(result: pd.DataFrame) -> pd.DataFrame:
    base = [c for c in ("경기날짜", "홈팀", "방문팀", "구장") if c in result.columns]
    algo = [c for c in _ALGO_PRED_COLS if c in result.columns]
    cols = list(base)
    if TARGET in result.columns:
        cols.append(TARGET)
    cols.extend(algo)
    if "예측_관중수" in result.columns:
        cols.append("예측_관중수")
    if "오차" in result.columns:
        cols.append("오차")
    out = result[cols].copy()

    rename: dict[str, str] = {}
    if TARGET in out.columns:
        rename[TARGET] = "실제 관중수(명)"
    for ac in algo:
        rename[ac] = _ALGO_PRED_DISPLAY.get(ac, ac)
    if "예측_관중수" in out.columns:
        rename["예측_관중수"] = (
            "예상 관중수(평균)(명)" if len(algo) > 1 else "예상 관중수(명)"
        )
    out = out.rename(columns=rename)

    if "실제 관중수(명)" in out.columns:
        out["실제 관중수(명)"] = _parse_attendance_series(out["실제 관중수(명)"])
    for c in out.columns:
        if c.endswith("(명)"):
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


def _parse_attendance_series(s: pd.Series) -> pd.Series:
    return parse_attendance_column(s)


def _normalize_game_date(value) -> pd.Timestamp | None:
    gdt = pd.to_datetime(value, errors="coerce")
    if pd.isna(gdt):
        return None
    return pd.Timestamp(gdt).normalize()


def _lookup_actual_attendance(attendance_df: pd.DataFrame, row: pd.Series) -> float:
    gdt = _normalize_game_date(row.get("경기날짜"))
    if gdt is None:
        return np.nan
    home = str(row["홈팀"]).strip()
    away = str(row["방문팀"]).strip()
    st_col = str(row.get("구장", "")).strip()

    sub = attendance_df.copy()
    sub["_gd"] = pd.to_datetime(sub["경기날짜"], errors="coerce").dt.normalize()
    mask = (
        (sub["_gd"] == gdt)
        & (sub["홈팀"].astype(str).str.strip() == home)
        & (sub["방문팀"].astype(str).str.strip() == away)
    )
    if st_col and "구장" in sub.columns:
        from common.stadium_aliases import STADIUM_ALIAS

        st_ok = {st_col, STADIUM_ALIAS.get(st_col, st_col)}
        mask &= sub["구장"].astype(str).str.strip().isin(st_ok)
    hit = sub.loc[mask]
    if hit.empty:
        return np.nan
    return float(_parse_attendance_series(hit[TARGET]).iloc[-1])


def _attendance_lookup_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    need = {"경기날짜", "홈팀", "방문팀", TARGET}
    if not need.issubset(df.columns):
        return None
    cols = list(need)
    if "구장" in df.columns:
        cols.append("구장")
    out = df[cols].copy()
    out[TARGET] = _parse_attendance_series(out[TARGET])
    out["경기날짜"] = pd.to_datetime(out["경기날짜"], errors="coerce")
    return out.dropna(subset=["경기날짜", TARGET])


def _load_local_attendance_csvs(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    patterns = (
        "data/interim/kbo_*attendance*.csv",
        "data/raw/kbo_*attendance*.csv",
        "data/kbo_*attendance*.csv",
    )
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig")
            except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                continue
            part = _attendance_lookup_columns(raw)
            if part is not None and not part.empty:
                frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_external_schedule_attendance(root: Path) -> pd.DataFrame:
    """data/external 일정·관중 TSV/CSV (헤더 없는 6열 형식 등)."""
    ext = root / "data" / "external"
    if not ext.is_dir():
        return pd.DataFrame()
    skip_names = {
        "batch_predict_schedule_template.csv",
        "kbo_stadium_info.csv",
    }
    frames: list[pd.DataFrame] = []
    for path in sorted(ext.iterdir()):
        if not path.is_file() or path.name in skip_names:
            continue
        if path.suffix.lower() not in {".tsv", ".txt", ".csv"}:
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        text: str | None = None
        for enc in ("utf-8-sig", "cp949", "utf-8"):
            try:
                text = blob.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            continue
        first = text.split("\n", 1)[0] if text else ""
        sep = "\t" if "\t" in first else ","
        try:
            raw = pd.read_csv(
                pd.io.common.StringIO(text),
                sep=sep,
                header=None,
                skipinitialspace=True,
            )
        except (pd.errors.ParserError, ValueError):
            continue
        if raw.shape[1] < 6:
            try:
                named = pd.read_csv(pd.io.common.StringIO(text), sep=sep, encoding="utf-8-sig")
            except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                continue
            part = _attendance_lookup_columns(named)
            if part is not None and not part.empty:
                frames.append(part)
            continue
        part = pd.DataFrame(
            {
                "경기날짜": raw.iloc[:, 0],
                "홈팀": raw.iloc[:, 2],
                "방문팀": raw.iloc[:, 3],
                "구장": raw.iloc[:, 4],
                TARGET: raw.iloc[:, 5],
            }
        )
        part = _attendance_lookup_columns(part)
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_attendance_lookup_df(
    attendance_df: pd.DataFrame | None,
    root: Path | None = None,
) -> pd.DataFrame:
    """앱 CSV·로컬 관중 파일·external 일정 TSV를 합친 실제 관중 조회용 테이블."""
    root = root or PROJECT_ROOT
    frames: list[pd.DataFrame] = []
    if attendance_df is not None and not attendance_df.empty:
        part = _attendance_lookup_columns(attendance_df)
        if part is not None and not part.empty:
            frames.append(part)
    local = _load_local_attendance_csvs(root)
    if not local.empty:
        frames.append(local)
    external = _load_external_schedule_attendance(root)
    if not external.empty:
        frames.append(external)
    if not frames:
        return pd.DataFrame(columns=["경기날짜", "홈팀", "방문팀", TARGET])
    merged = pd.concat(frames, ignore_index=True)
    merged["_gd"] = pd.to_datetime(merged["경기날짜"], errors="coerce").dt.normalize()
    merged["홈팀"] = merged["홈팀"].astype(str).str.strip()
    merged["방문팀"] = merged["방문팀"].astype(str).str.strip()
    merged = merged.dropna(subset=["_gd", TARGET])
    merged = merged.sort_values("_gd")
    merged = merged.drop_duplicates(subset=["_gd", "홈팀", "방문팀"], keep="last")
    return merged.drop(columns=["_gd"]).reset_index(drop=True)


def _fill_actual_from_lookup(out: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    if lookup.empty:
        return out
    hist = lookup.copy()
    hist[TARGET] = _parse_attendance_series(hist[TARGET])
    for idx, r in out.iterrows():
        if pd.notna(out.at[idx, TARGET]):
            continue
        found = _lookup_actual_attendance(hist, r)
        if pd.notna(found):
            out.at[idx, TARGET] = found
    return out


def _recompute_batch_error(out: pd.DataFrame) -> pd.DataFrame:
    if "예측_관중수" not in out.columns or not out[TARGET].notna().any():
        return out
    ref = "예측_평균" if "예측_평균" in out.columns else "예측_관중수"
    if ref not in out.columns:
        return out
    out["오차"] = (
        pd.to_numeric(out[ref], errors="coerce") - out[TARGET]
    ).abs().round().astype("Int64")
    return out


def _batch_weather_session_key(
    default_temp: float,
    default_rain: float,
    default_hum: float,
) -> str:
    return f"{default_temp:.2f}|{default_rain:.2f}|{default_hum:.1f}"


def clear_kbo_attendance_cache() -> None:
    _kbo_graph_daily_attendance.clear()


@st.cache_data(
    ttl=_KBO_BATCH_ATT_TTL,
    show_spinner="KBO 기록실에서 실제 관중을 조회하는 중…",
)
def _kbo_graph_daily_attendance(years: tuple[int, ...], data_fp: str) -> pd.DataFrame:
    from data_collection.kbo_scraping import (
        enrich_attendance_df,
        scrape_kbo_attendance,
        scraped_rows_to_dataframe,
    )

    headless = os.environ.get("KBO_SCRAPE_HEADLESS", "1") != "0"
    raw = scrape_kbo_attendance(list(years), headless=headless)
    frames: list[pd.DataFrame] = []
    for year in sorted(set(int(y) for y in years)):
        rows = raw.get(year) or []
        if not rows:
            continue
        raw_df = scraped_rows_to_dataframe(rows)
        if raw_df.empty:
            continue
        enriched = enrich_attendance_df(raw_df, year)
        part = _attendance_lookup_columns(enriched)
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def enrich_batch_actual_attendance(
    result: pd.DataFrame,
    attendance_df: pd.DataFrame | None,
    *,
    project_root: Path | None = None,
    try_kbo_scrape: bool = False,
) -> pd.DataFrame:
    out = result.copy()
    if TARGET not in out.columns:
        out[TARGET] = np.nan
    out[TARGET] = _parse_attendance_series(out[TARGET])

    lookup = build_attendance_lookup_df(attendance_df, project_root)
    out = _fill_actual_from_lookup(out, lookup)

    if try_kbo_scrape:
        today = pd.Timestamp.now().normalize()
        years: set[int] = set()
        for _, r in out.iterrows():
            if pd.notna(r.get(TARGET)):
                continue
            gdt = _normalize_game_date(r.get("경기날짜"))
            if gdt is None or gdt >= today:
                continue
            years.add(int(gdt.year))
        if years:
            data_fp = attendance_sources_fingerprint(project_root or PROJECT_ROOT)
            scraped = _kbo_graph_daily_attendance(tuple(sorted(years)), data_fp)
            if not scraped.empty:
                lookup = pd.concat([lookup, scraped], ignore_index=True)
                lookup = build_attendance_lookup_df(lookup, root=None)
                out = _fill_actual_from_lookup(out, lookup)

    return _recompute_batch_error(out)


def _build_batch_compare_df(result: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    chart = result.copy()
    chart["_dt"] = pd.to_datetime(chart["경기날짜"], errors="coerce")
    chart = chart.sort_values("_dt", na_position="last")
    for _, r in chart.iterrows():
        pred = pd.to_numeric(r.get("예측_관중수"), errors="coerce")
        if pd.isna(pred):
            continue
        gdt = r["_dt"]
        home = str(r["홈팀"]).strip()
        away = str(r["방문팀"]).strip()
        d = pd.Timestamp(gdt).strftime("%m/%d") if pd.notna(gdt) else "?"
        actual = np.nan
        if TARGET in r.index:
            actual = pd.to_numeric(r[TARGET], errors="coerce")
        has_actual = pd.notna(actual)
        rows.append(
            {
                "경기": f"{d}\n{home} vs {away}",
                "실제": float(actual) if has_actual else np.nan,
                "예측": float(pred),
                "예측_모델": _algo_preds_from_row(r),
                "has_actual": bool(has_actual),
            }
        )
    return pd.DataFrame(rows)


def _batch_slot_bars(row: pd.Series) -> list[tuple[str, int, str]]:
    """한 경기 슬롯 막대: 실제(있으면) + 알고리즘별 예측."""
    bars: list[tuple[str, int, str]] = []
    if row["has_actual"] and pd.notna(row["실제"]):
        bars.append(("실제(기록)", int(row["실제"]), _ACTUAL_BAR_COLOR))
    items = _ordered_batch_algo_preds(row.get("예측_모델") or {})
    if items:
        if len(items) == 1:
            label, val = items[0]
            bars.append((label, val, _PRED_BAR_COLOR_SINGLE))
        else:
            for label, val in items:
                bars.append((label, val, _model_bar_color(label)))
    elif pd.notna(row.get("예측")):
        pred = int(row["예측"])
        if row["has_actual"]:
            bars.append(("예측(ML 평균)", pred, _PRED_BAR_COLOR_SINGLE))
        else:
            bars.append(("예정 경기 예측", pred, "#18e6ff"))
    return bars


def _plot_batch_actual_vs_predicted(result: pd.DataFrame) -> None:
    compare = _build_batch_compare_df(result)
    if compare.empty:
        return

    n = len(compare)
    max_bars = max(len(_batch_slot_bars(compare.iloc[i])) for i in range(n))
    slot_w = 2.35 if max_bars >= 4 else 2.0 if max_bars >= 3 else 1.7
    fig_w = min(18.0, max(10.0, n * slot_w + 2))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2), dpi=120)
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")

    labels = compare["경기"].tolist()
    x = np.arange(n)
    legend_seen: set[str] = set()
    slot_drawn: list[tuple[float, list[tuple[str, int, str]], list[float]]] = []
    ymax = 1.0

    for i in range(n):
        row = compare.iloc[i]
        bars = _batch_slot_bars(row)
        y_slot, offs = _draw_bars_at_slot(ax, x[i], bars, legend_seen=legend_seen)
        ymax = max(ymax, y_slot)
        slot_drawn.append((x[i], bars, offs))

    dy = max(ymax * 0.015, 180.0)
    val_fs = 9 if max_bars >= 4 else 10 if max_bars >= 3 else 11
    for xi, bars, offs in slot_drawn:
        for off, (label, val, _color) in zip(offs, bars):
            ax.text(
                xi + off,
                val + dy,
                f"{int(val):,}",
                ha="center",
                color="#e8eef5",
                fontsize=val_fs,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=8)
    ax.set_ylabel("관중 수")
    ax.set_title("경기별 실제 vs 예측 관중", color="white")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    n_legend = len(legend_seen)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(5, max(2, n_legend)),
        fontsize=9,
        frameon=True,
        facecolor="#0d1a2b",
        edgecolor="#9fb3c8",
        labelcolor="white",
    )
    for spine in ax.spines.values():
        spine.set_color("#9fb3c8")
    plt.yticks(color="white")
    fig.subplots_adjust(bottom=0.30 if n_legend >= 4 else 0.26)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _batch_session_active() -> bool:
    return (
        "batch_attendance_result" in st.session_state
        and bool(st.session_state.get("batch_upload_sig"))
    )


def clear_batch_schedule_session() -> None:
    for key in (
        "batch_attendance_result",
        "batch_schedule_df",
        "batch_upload_sig",
        "batch_ml_labels",
        "batch_ml_models_key",
        "batch_data_fp",
        "batch_weather_key",
        "batch_schedule_source",
        "sidebar_schedule_csv",
        "_csv_apply_error",
    ):
        st.session_state.pop(key, None)


def maybe_refresh_batch_predictions(
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
    cap_map: dict[str, int],
    ml_train_ok: bool,
) -> bool:
    """저장된 일정 CSV로 학습·관중 파일·날씨·모델 변경 시 예측을 다시 계산."""
    sched = st.session_state.get("batch_schedule_df")
    if sched is None or not isinstance(sched, pd.DataFrame) or sched.empty:
        return False
    if not ml_train_ok or not chosen:
        return False

    fp = attendance_sources_fingerprint(PROJECT_ROOT)
    wkey = _batch_weather_session_key(default_temp, default_rain, default_hum)
    ml_key = _ml_models_session_key(chosen)
    if (
        st.session_state.get("batch_data_fp") == fp
        and st.session_state.get("batch_weather_key") == wkey
        and st.session_state.get("batch_ml_models_key") == ml_key
        and "batch_attendance_result" in st.session_state
    ):
        return False

    try:
        result = _run_predictions(
            sched,
            chosen,
            default_temp=default_temp,
            default_rain=default_rain,
            default_hum=default_hum,
            cap_map=cap_map,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        st.session_state["_csv_apply_error"] = str(e)
        return False
    except Exception as e:
        st.session_state["_csv_apply_error"] = f"예측 실패: {e}"
        return False

    if st.session_state.get("batch_data_fp") != fp:
        clear_kbo_attendance_cache()
    st.session_state["batch_attendance_result"] = result
    st.session_state["batch_data_fp"] = fp
    st.session_state["batch_weather_key"] = wkey
    st.session_state["batch_ml_models_key"] = ml_key
    st.session_state["batch_ml_labels"] = ", ".join(chosen)
    st.session_state.pop("_csv_apply_error", None)
    return True


def apply_uploaded_schedule_csv(
    uploaded,
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
    cap_by_stadium: dict[str, int],
    ml_train_ok: bool,
) -> tuple[bool, str]:
    if uploaded is None:
        return False, "먼저 경기 일정 CSV를 올려 주세요."
    if not ml_train_ok:
        return False, "`kbo_train_ready.csv` 없음 — `build_features.py` 실행이 필요합니다."
    if not chosen:
        return False, "ML 알고리즘을 하나 이상 선택해 주세요."

    sig = f"{uploaded.name}:{uploaded.size}"
    try:
        sched = _read_uploaded_schedule(uploaded)
        result = _run_predictions(
            sched,
            chosen,
            default_temp=default_temp,
            default_rain=default_rain,
            default_hum=default_hum,
            cap_map=cap_by_stadium,
        )
        st.session_state["batch_attendance_result"] = result
        st.session_state["batch_schedule_df"] = sched.copy()
        st.session_state["batch_upload_sig"] = sig
        st.session_state["batch_ml_labels"] = ", ".join(chosen)
        st.session_state["batch_ml_models_key"] = _ml_models_session_key(chosen)
        st.session_state["batch_data_fp"] = attendance_sources_fingerprint(PROJECT_ROOT)
        st.session_state["batch_weather_key"] = _batch_weather_session_key(
            default_temp, default_rain, default_hum
        )
        st.session_state["batch_schedule_source"] = str(uploaded.name)
        return True, ""
    except (KeyError, ValueError, FileNotFoundError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"예측 실패: {e}"


def on_sidebar_schedule_upload_cleared() -> None:
    clear_batch_schedule_session()


def _batch_predictions_up_to_date(
    uploaded_sig: str,
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
) -> bool:
    """업로드·모델·날씨·학습/관중 파일이 모두 현재 설정과 일치하는지."""
    if "batch_attendance_result" not in st.session_state:
        return False
    if st.session_state.get("batch_upload_sig") != uploaded_sig:
        return False
    if st.session_state.get("batch_ml_models_key") != _ml_models_session_key(chosen):
        return False
    wkey = _batch_weather_session_key(default_temp, default_rain, default_hum)
    if st.session_state.get("batch_weather_key") != wkey:
        return False
    if st.session_state.get("batch_data_fp") != attendance_sources_fingerprint(PROJECT_ROOT):
        return False
    return True


def try_auto_apply_schedule_csv(
    uploaded,
    chosen: list[str],
    *,
    default_temp: float,
    default_rain: float,
    default_hum: float,
    cap_by_stadium: dict[str, int],
    ml_train_ok: bool,
) -> str | None:
    """CSV 선택·변경 시 자동 예측. 오류 메시지 또는 None(성공·스킵)."""
    if uploaded is None:
        return None
    sig = f"{uploaded.name}:{uploaded.size}"
    if _batch_predictions_up_to_date(
        sig,
        chosen,
        default_temp=default_temp,
        default_rain=default_rain,
        default_hum=default_hum,
    ):
        return None
    ok, err = apply_uploaded_schedule_csv(
        uploaded,
        chosen,
        default_temp=default_temp,
        default_rain=default_rain,
        default_hum=default_hum,
        cap_by_stadium=cap_by_stadium,
        ml_train_ok=ml_train_ok,
    )
    return None if ok else err


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
    if _batch_predictions_up_to_date(
        sig,
        chosen,
        default_temp=default_temp,
        default_rain=default_rain,
        default_hum=default_hum,
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
        st.session_state["batch_schedule_df"] = sched.copy()
        st.session_state["batch_upload_sig"] = sig
        st.session_state["batch_ml_labels"] = ", ".join(chosen)
        st.session_state["batch_ml_models_key"] = _ml_models_session_key(chosen)
        st.session_state["batch_data_fp"] = attendance_sources_fingerprint(PROJECT_ROOT)
        st.session_state["batch_weather_key"] = _batch_weather_session_key(
            default_temp, default_rain, default_hum
        )
        return True
    except (KeyError, ValueError, FileNotFoundError) as e:
        st.sidebar.error(str(e))
        return False
    except Exception as e:
        st.sidebar.error(f"예측 실패: {e}")
        return False


def _merge_single_game_result(
    result: pd.DataFrame,
    single_game_row: pd.DataFrame | None,
) -> pd.DataFrame:
    """CSV 일괄 결과 + 사이드바 단일 경기 결과를 통합."""
    if single_game_row is None or single_game_row.empty:
        return result
    merged = pd.concat([single_game_row.copy(), result.copy()], ignore_index=True, sort=False)
    keys = ["경기날짜", "홈팀", "방문팀", "구장"]
    keys = [k for k in keys if k in merged.columns]
    if keys:
        merged = merged.drop_duplicates(subset=keys, keep="first")
    return merged.reset_index(drop=True)


def render_csv_batch_results_main(
    *,
    chosen: list[str],
    cap_by_stadium: dict[str, int],
    ml_train_ok: bool,
    attendance_df: pd.DataFrame | None = None,
    embedded: bool = False,
    try_kbo_scrape: bool = False,
    single_game_row: pd.DataFrame | None = None,
) -> None:
    """메인 영역 —  예측 결과."""

    if not embedded:
        st.markdown(
            '<div class="main-title">📈 KBO 관람 수요 예측 시스템</div>',
            unsafe_allow_html=True,
        )
    if not embedded:
        _src = st.session_state.get("batch_schedule_source", "업로드한 일정")
        st.markdown(
            '<div class="sub-text">'
            f"<b>{html.escape(str(_src))}</b> 기준 경기별 예측입니다."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
    elif _batch_session_active():
        _src = st.session_state.get("batch_schedule_source", "업로드한 일정")
        st.caption(f"일정 파일: **{html.escape(str(_src))}**")

    if not ml_train_ok:
        st.error(
            "`data/processed/kbo_train_ready.csv` 없음 — "
            "먼저 `python3 scripts/features/build_features.py` 를 실행하세요."
        )
        return

    if "batch_attendance_result" not in st.session_state:
        return

    force_kbo = bool(st.session_state.pop("batch_force_kbo_refresh", False))
    if force_kbo:
        clear_kbo_attendance_cache()

    base_result = _merge_single_game_result(
        st.session_state["batch_attendance_result"].copy(),
        single_game_row,
    )
    result = enrich_batch_actual_attendance(
        base_result,
        attendance_df,
        project_root=PROJECT_ROOT,
        try_kbo_scrape=try_kbo_scrape or force_kbo,
    )
    if "예측_관중수" not in result.columns:
        return

    avg_att = int(round(float(result["예측_관중수"].mean())))
    n_games = len(result)
    congestion = _avg_congestion_pct(result, cap_by_stadium)
    _plan = classify_congestion_pct(congestion)

    st.markdown("## 📊 예측 요약")

    _ml_labels = st.session_state.get("batch_ml_labels", ", ".join(chosen))
    st.caption(f"모델: **{_ml_labels}** (경기별 평균 반영)")

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
                f"이번 CSV **{n_games:,}경기** 예측의 평균 **{avg_att:,}명**은"
            ),
            selectbox_key="batch_ml_feat_imp_model",
        )

    _chart_hdr, _chart_btn = st.columns([5, 1])
    with _chart_hdr:
        st.markdown("## 📊 경기별 실제 관중 vs 예측 관중")
    with _chart_btn:
        if st.button(
            "실제 관중 새로고침",
            use_container_width=True,
            key="btn_refresh_batch_actual",
            help="로컬·external 관중 파일 변경·KBO 기록실 재조회",
        ):
            clear_kbo_attendance_cache()
            st.session_state["batch_force_kbo_refresh"] = True
            st.rerun()
    _plot_batch_actual_vs_predicted(result)

    _algo_in_result = [c for c in _ALGO_PRED_COLS if c in result.columns]
    with st.expander("📋 경기별 상세", expanded=False):
        if len(_algo_in_result) > 1:
            st.caption(f"알고리즘별 예측값과 평균 예측값을 함께 표시합니다.")
        elif len(_algo_in_result) == 1:
            st.caption(f"예측 모델: **{_algo_in_result[0].replace('예측_', '')}**")
        st.dataframe(_format_result_table(result), hide_index=True, use_container_width=True)

    if TARGET in result.columns and "오차" in result.columns and result["오차"].notna().any():
        st.metric("실제 관중 대비 평균 오차 (MAE)", f"{float(result['오차'].mean()):,.0f} 명")



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

    result = enrich_batch_actual_attendance(
        st.session_state["batch_attendance_result"].copy(),
        None,
        project_root=PROJECT_ROOT,
        try_kbo_scrape=os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "").strip().lower()
        != "cloud",
    )
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

    st.markdown("## 📊 경기별 실제 vs 예측 관중")
    _plot_batch_actual_vs_predicted(result)

    with st.expander("📋 경기별 상세", expanded=False):
        st.dataframe(_format_result_table(result), hide_index=True, use_container_width=True)

    if TARGET in result.columns and "오차" in result.columns and result["오차"].notna().any():
        st.metric("실제 관중 대비 평균 오차 (MAE)", f"{float(result['오차'].mean()):,.0f} 명")

