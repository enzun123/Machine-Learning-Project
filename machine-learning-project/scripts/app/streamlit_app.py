# streamlit_app.py

import hashlib
import html
import logging
import os
import sys
from pathlib import Path

# Streamlit Cloud: common.* 는 scripts/ 패키지 — third-party·common import 전에 path 등록
_SCRIPTS = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.theme import (
    actual_bar_color,
    add_chart_legend,
    apply_axes_theme,
    apply_figure_theme,
    finalize_figure_for_streamlit,
    model_bar_color,
    model_bar_colors,
    pred_single_bar_color,
)
from app.theme_watcher import mount_theme_watcher
from common.congestion_levels import classify_congestion_pct
from common.kma_vilage_fcst import redact_api_secrets
from common.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

_NANUM_GOTHIC_URL = (
    "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
)


def _is_streamlit_cloud() -> bool:
    return os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "").strip().lower() == "cloud"


def _default_web_recent_enabled() -> bool:
    env = os.environ.get("STREAMLIT_WEB_RECENT")
    if env is not None:
        return env.strip() != "0"
    if _is_streamlit_cloud():
        return False
    return True


@st.cache_resource
def _ensure_korean_matplotlib_font() -> str:
    """Linux(Streamlit Cloud) 등에서 한글 깨짐 방지 — Nanum Gothic 등록."""
    plt.rcParams["axes.unicode_minus"] = False
    fonts_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    font_path = fonts_dir / "NanumGothic.ttf"

    if not font_path.is_file():
        try:
            import urllib.request

            urllib.request.urlretrieve(_NANUM_GOTHIC_URL, font_path)
        except Exception as e:
            logger.warning("Nanum Gothic 다운로드 실패: %s", e)

    if font_path.is_file():
        try:
            fm.fontManager.addfont(str(font_path))
            family = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = family
            return family
        except Exception as e:
            logger.warning("Nanum Gothic 등록 실패: %s", e)

    _font_names = {f.name for f in fm.fontManager.ttflist}
    for _fam in (
        "NanumGothic",
        "Nanum Gothic",
        "AppleGothic",
        "Apple SD Gothic Neo",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ):
        if _fam in _font_names:
            plt.rcParams["font.family"] = _fam
            return _fam

    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="KBO 관람 수요 예측 시스템",
    layout="wide",
)

mount_theme_watcher()
_ensure_korean_matplotlib_font()

# =========================
# 데이터 불러오기
# =========================
@st.cache_data
def load_data(_data_fp: str):

    BASE_DIR = Path(__file__).resolve().parent
    root = BASE_DIR.parent.parent

    possible_paths = [

        # 현재 폴더
        BASE_DIR / "kbo_2025_attendance_weather.csv",

        # 상위 폴더들
        BASE_DIR.parent / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent.parent / "kbo_2025_attendance_weather.csv",

        # data 폴더
        root / "data" / "kbo_2025_attendance_weather.csv",
        root / "data" / "interim" / "kbo_2025_attendance_weather.csv",
        BASE_DIR.parent.parent.parent / "data" / "kbo_2025_attendance_weather.csv",
    ]

    DATA_PATH = None

    for path in possible_paths:

        if path.exists():
            DATA_PATH = path
            break

    if DATA_PATH is None:

        st.error(
            "CSV 파일을 찾을 수 없습니다.\n"
            "kbo_2025_attendance_weather.csv 파일을 "
            "프로젝트 폴더 또는 data 폴더에 넣어주세요."
        )

        st.stop()

    df = pd.read_csv(DATA_PATH)

    _need = {"경기날짜", "홈팀", "방문팀", "구장", "관중수"}
    if not _need.issubset(df.columns):
        missing = sorted(_need - set(df.columns))
        st.error(
            "CSV에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )
        st.stop()

    df["경기날짜"] = pd.to_datetime(
        df["경기날짜"],
        errors="coerce"
    )

    df["일합계강수량(mm)"] = (
        df["일합계강수량(mm)"]
        .fillna(0)
    )

    return df


def _default_home_team_for_stadium(stadium_name: str, attendance_df: pd.DataFrame) -> str:
    """구장 변경 시 기본 홈팀. 잠실은 LG·두산 공유 → LG 우선."""
    st_key = str(stadium_name).strip()
    homes = sorted(attendance_df["홈팀"].dropna().unique())
    explicit = {
        "잠실": "LG",
        "고척": "키움",
        "수원": "KT",
        "문학": "SSG",
        "인천": "SSG",
        "광주": "KIA",
        "대구": "삼성",
        "창원": "NC",
        "한밭": "한화",
        "대전": "한화",
        "사직": "롯데",
        "부산": "롯데",
        "청주": "한화",
        "포항": "삼성",
    }
    if st_key in explicit:
        t = explicit[st_key]
        if t in homes:
            return t
    sub = attendance_df.loc[attendance_df["구장"] == st_key, "홈팀"].dropna()
    if len(sub) > 0:
        m = sub.mode()
        if len(m) > 0:
            v = str(m.iloc[0])
            if v in homes:
                return v
    return homes[0] if homes else ""


def _fallback_stadium_capacity() -> dict[str, int]:
    """CSV를 못 읽을 때만 사용 (kbo_stadium_info.csv와 동기 유지 권장)."""
    return {
        "잠실": 23750,
        "고척": 16744,
        "인천": 23000,
        "문학": 23000,
        "사직": 22669,
        "부산": 22669,
        "창원": 22112,
        "울산": 15000,
        "청주": 10500,
        "포항": 15000,
        "광주": 22000,
        "대전": 17000,
        "한밭": 17000,
        "대구": 24000,
        "수원": 18700,
    }


def _load_app_capacity_map() -> dict[str, int]:
    from common.stadium_capacity import load_capacity_map_for_app

    out = load_capacity_map_for_app(PROJECT_ROOT)
    return out if out else _fallback_stadium_capacity()


def _ensure_session_data() -> None:
    from common.attendance_parse import attendance_sources_fingerprint

    fp = attendance_sources_fingerprint(PROJECT_ROOT)
    if st.session_state.get("df_data_fp") != fp or "df" not in st.session_state:
        st.session_state.df = load_data(fp)
        st.session_state.df_data_fp = fp
    # kbo_stadium_info / kbo_train_ready 갱신 시 정원 반영 (세션에 예전 값 고정 방지)
    st.session_state.cap_by_stadium = _load_app_capacity_map()


def get_capacity(stadium_name: str, home_team: str | None = None) -> int:
    from common.stadium_capacity import venue_clip_capacity

    cap_by = st.session_state.get("cap_by_stadium") or _load_app_capacity_map()
    home = home_team if home_team is not None else st.session_state.get("fld_home_team", "")
    return venue_clip_capacity(str(stadium_name), str(home), cap_by)


try:
    _KBO_RECENT_TTL = int(os.environ.get("KBO_APP_RECENT_TTL_SEC", "900"))
except ValueError:
    _KBO_RECENT_TTL = 900


@st.cache_data(
    ttl=_KBO_RECENT_TTL,
    show_spinner="KBO 기록실에서 최근 경기를 불러오는 중…",
)
def _recent_five_kbo(stadium: str, before_iso: str) -> pd.DataFrame:
    try:
        from data_collection.fetch_recent_crowd import fetch_recent_games

        headless = os.environ.get("KBO_SCRAPE_HEADLESS", "1") != "0"
        out = fetch_recent_games(
            stadium,
            n=5,
            before=before_iso,
            headless=headless,
        )
        return out if out is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(
            "KBO 최근 경기 스크랩 실패 (stadium=%s, before=%s): %s",
            stadium,
            before_iso,
            e,
            exc_info=True,
        )
        return pd.DataFrame()


def _weather_for_historical_row(attendance_df: pd.DataFrame, row: pd.Series) -> tuple[float, float, float, float]:
    """과거 경기 행에 맞는 기상(로컬 CSV 병합, 없으면 기본값)."""
    from common.config import (
        DEFAULT_RH_MEDIAN_FALLBACK,
        DEFAULT_TEMP_MEDIAN_FALLBACK,
        DEFAULT_WIND_MEDIAN_FALLBACK,
    )

    temp = DEFAULT_TEMP_MEDIAN_FALLBACK
    rain = 0.0
    hum = DEFAULT_RH_MEDIAN_FALLBACK
    wind = DEFAULT_WIND_MEDIAN_FALLBACK
    if attendance_df is None or attendance_df.empty:
        return temp, rain, hum, wind

    gdt = pd.Timestamp(row["경기날짜"]).normalize()
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
    if st_col:
        mask &= sub["구장"].astype(str).str.strip() == st_col
    hit = sub.loc[mask]
    if hit.empty:
        return temp, rain, hum, wind
    r = hit.iloc[-1]

    def _f(col: str, default: float) -> float:
        if col in r.index and pd.notna(r[col]):
            return float(r[col])
        return default

    return (
        _f("일평균기온(°C)", temp),
        _f("일합계강수량(mm)", rain),
        _f("일평균상대습도(%)", hum),
        _f("일평균풍속(m/s)", wind),
    )


def _predict_ml_attendance_by_model_for_row(
    tr: pd.DataFrame,
    *,
    home: str,
    away: str,
    stadium_name: str,
    game_date,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
    wind_mps: float,
    cap_map: dict[str, int],
    ml_choice_map: dict[str, bool],
) -> dict[str, int]:
    """단일 과거 경기 — 선택 모델별 예측."""
    from common.stadium_capacity import venue_clip_capacity
    from modeling.batch_feature_builder import build_ml_feature_dataframe

    cap_hi = max(1, venue_clip_capacity(stadium_name, home, cap_map))
    out: dict[str, int] = {}
    try:
        X = build_ml_feature_dataframe(
            tr,
            home=home,
            away=away,
            stadium=stadium_name,
            game_date=game_date,
            temp_c=temp_c,
            rain_mm=rain_mm,
            hum_pct=hum_pct,
            cap_map=cap_map,
            wind_mps=wind_mps,
        )
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].astype(str).fillna("missing")
        for key, label, fname in ML_MODEL_REGISTRY:
            if not ml_choice_map.get(key):
                continue
            pipe = _load_ml_pipeline(fname)
            if pipe is None:
                continue
            raw = float(np.asarray(pipe.predict(X))[0])
            out[label] = min(int(round(max(0.0, raw))), cap_hi)
    except Exception as e:
        logger.warning("과거 경기 ML 예측 실패 (%s vs %s): %s", home, away, e)
        return {}
    return out


def _predict_ml_attendance_for_row(
    tr: pd.DataFrame,
    *,
    home: str,
    away: str,
    stadium_name: str,
    game_date,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
    wind_mps: float,
    cap_map: dict[str, int],
    ml_choice_map: dict[str, bool],
) -> int | None:
    """단일 과거 경기 ML 예측(선택 모델 평균)."""
    by_model = _predict_ml_attendance_by_model_for_row(
        tr,
        home=home,
        away=away,
        stadium_name=stadium_name,
        game_date=game_date,
        temp_c=temp_c,
        rain_mm=rain_mm,
        hum_pct=hum_pct,
        wind_mps=wind_mps,
        cap_map=cap_map,
        ml_choice_map=ml_choice_map,
    )
    if not by_model:
        return None
    return int(round(float(np.mean(list(by_model.values())))))


def _selected_match_chart_label(
    game_date: object,
    home_team: str,
    away_team: str,
) -> str:
    d = pd.Timestamp(game_date).strftime("%m/%d")
    return f"{d}\n{home_team} vs {away_team}\n선택한 경기 예측"


def _resolve_selected_game_actual(
    attendance_df: pd.DataFrame,
    game_date: object,
    home_team: str,
    away_team: str,
    stadium: str,
    *,
    try_kbo: bool,
) -> int | None:
    """선택 경기의 실제 관중(로컬·external·KBO 기록실). 없으면 None."""
    from app.csv_batch_predict_ui import (
        _kbo_graph_daily_attendance,
        _lookup_actual_attendance,
        build_attendance_lookup_df,
    )
    from common.attendance_parse import attendance_sources_fingerprint

    row = pd.Series(
        {
            "경기날짜": pd.Timestamp(game_date).strftime("%Y-%m-%d"),
            "홈팀": home_team,
            "방문팀": away_team,
            "구장": stadium,
        }
    )
    lookup = build_attendance_lookup_df(attendance_df, PROJECT_ROOT)
    found = _lookup_actual_attendance(lookup, row)
    gdt = pd.Timestamp(game_date).normalize()
    today = pd.Timestamp.now().normalize()

    if pd.isna(found) and try_kbo and gdt <= today:
        data_fp = attendance_sources_fingerprint(PROJECT_ROOT)
        scraped = _kbo_graph_daily_attendance((int(gdt.year),), data_fp)
        if not scraped.empty:
            lookup = build_attendance_lookup_df(
                pd.concat([lookup, scraped], ignore_index=True)
                if not lookup.empty
                else scraped,
                PROJECT_ROOT,
            )
            found = _lookup_actual_attendance(lookup, row)

    if pd.isna(found) and try_kbo and gdt < today:
        try:
            from data_collection.fetch_recent_crowd import fetch_recent_games

            headless = os.environ.get("KBO_SCRAPE_HEADLESS", "1") != "0"
            before_next = (gdt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            recent = fetch_recent_games(
                stadium,
                n=40,
                before=before_next,
                headless=headless,
            )
            if not recent.empty:
                recent = recent.copy()
                recent["_gd"] = pd.to_datetime(
                    recent["경기날짜"], errors="coerce"
                ).dt.normalize()
                hit = recent.loc[
                    (recent["_gd"] == gdt)
                    & (recent["홈팀"].astype(str).str.strip() == str(home_team).strip())
                    & (
                        recent["방문팀"].astype(str).str.strip()
                        == str(away_team).strip()
                    )
                ]
                if not hit.empty:
                    val = pd.to_numeric(hit["관중수"], errors="coerce").dropna()
                    if len(val):
                        found = float(val.iloc[-1])
        except Exception as e:
            logger.warning("선택 경기 KBO 실제 관중 조회 실패: %s", e)

    if pd.isna(found):
        return None
    return int(round(float(found)))


def _ordered_ml_predictions(preds: dict[str, int]) -> list[tuple[str, int]]:
    order = [label for _k, label, _fname in ML_MODEL_REGISTRY if label in preds]
    return [(label, int(preds[label])) for label in order]


def _slot_bar_layout(n_bars: int, *, span: float = 0.9, width_ratio: float = 0.72) -> tuple[list[float], float]:
    """슬롯 중심 기준 막대 중심 오프셋·막대 너비 (겹침 방지용 간격 포함)."""
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
    """한 경기 슬롯에 막대 그룹. (ymax, 각 막대 중심 오프셋) 반환."""
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


def _plot_recent_actual_vs_predicted(
    compare: pd.DataFrame,
    *,
    future_pred: int,
    stadium_name: str,
    game_date: object | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    future_preds: dict[str, int] | None = None,
    selected_actual: int | None = None,
) -> None:
    """최근 N경기 실제·예측 막대 + 선택 경기(알고리즘별 막대 또는 단일 예측)."""
    n = len(compare)
    future_items = _ordered_ml_predictions(future_preds) if future_preds else []
    n_slots = n + 1
    if "예측_모델" in compare.columns:
        slot_models = [
            v if isinstance(v, dict) else {}
            for v in compare["예측_모델"].tolist()
        ]
    else:
        slot_models = [{}] * n

    max_bars_per_slot = 1
    for i in range(n):
        items = _ordered_ml_predictions(slot_models[i])
        max_bars_per_slot = max(max_bars_per_slot, 1 + len(items) if items else 2)
    has_selected_actual = selected_actual is not None
    _fut_bar_n = len(future_items) if future_items else 1
    if has_selected_actual:
        _fut_bar_n += 1
    max_bars_per_slot = max(max_bars_per_slot, _fut_bar_n)
    slot_w = 2.35 if max_bars_per_slot >= 4 else 2.0 if max_bars_per_slot >= 3 else 1.9
    fig_w = min(18.0, max(12.0, n_slots * slot_w))
    fig_h = 7.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    th = apply_figure_theme(fig, ax)

    if game_date is not None and home_team and away_team:
        future_lbl = _selected_match_chart_label(game_date, home_team, away_team)
    else:
        future_lbl = "선택한 경기 예측"
    labels = compare["경기"].tolist() + [future_lbl]
    x = np.arange(len(labels))

    actuals = compare["실제"].to_numpy(dtype=float)
    preds = compare["예측"].to_numpy(dtype=float)

    def _hist_slot_bars(i: int) -> list[tuple[str, int, str]]:
        bars: list[tuple[str, int, str]] = [
            ("실제(기록실)", int(actuals[i]), actual_bar_color()),
        ]
        items = _ordered_ml_predictions(slot_models[i])
        if items:
            if len(items) == 1:
                label, val = items[0]
                bars.append((label, val, pred_single_bar_color()))
            else:
                for label, val in items:
                    bars.append((label, val, model_bar_color(label)))
        elif i < len(preds) and not np.isnan(preds[i]):
            bars.append(("예측(ML 평균)", int(preds[i]), pred_single_bar_color()))
        return bars

    def _future_slot_bars() -> list[tuple[str, int, str]]:
        bars: list[tuple[str, int, str]] = []
        if has_selected_actual:
            bars.append(("실제(기록실)", int(selected_actual), actual_bar_color()))
        if not future_items:
            if not bars:
                return [("이번 경기 예측", int(future_pred), th.accent_cyan)]
            bars.append(("예측(ML 평균)", int(future_pred), pred_single_bar_color()))
            return bars
        if len(future_items) == 1:
            label, val = future_items[0]
            bars.append((label, val, pred_single_bar_color()))
        else:
            for label, val in future_items:
                bars.append((label, val, model_bar_color(label)))
        return bars

    legend_seen: set[str] = set()
    slot_drawn: list[tuple[float, list[tuple[str, int, str]], list[float]]] = []
    ymax = max(
        float(np.nanmax(actuals)) if len(actuals) else 0.0,
        float(np.nanmax(preds)) if len(preds) and not np.all(np.isnan(preds)) else 0.0,
        float(future_pred),
        float(selected_actual) if has_selected_actual else 0.0,
        1.0,
    )
    for i in range(n):
        bars = _hist_slot_bars(i)
        y_slot, offs = _draw_bars_at_slot(ax, x[i], bars, legend_seen=legend_seen)
        ymax = max(ymax, y_slot)
        slot_drawn.append((x[i], bars, offs))

    future_bars = _future_slot_bars()
    y_fut, fut_offs = _draw_bars_at_slot(ax, x[n], future_bars, legend_seen=legend_seen)
    ymax = max(ymax, y_fut)

    dy = max(ymax * 0.018, 280.0)
    val_fs = 9 if max_bars_per_slot >= 4 else 10 if max_bars_per_slot >= 3 else 12
    for xi, bars, offs in slot_drawn:
        for off, (_label, val, _color) in zip(offs, bars):
            ax.text(
                xi + off,
                val + dy,
                f"{int(val):,}",
                ha="center",
                color=th.fg,
                fontsize=val_fs,
            )

    for off, (_label, val, _color) in zip(fut_offs, future_bars):
        ax.text(
            x[n] + off,
            val + dy,
            f"{int(val):,}",
            ha="center",
            color=th.fg,
            fontsize=val_fs,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=th.tick, fontsize=10)
    ax.set_ylabel("관중 수", fontsize=13)
    title_fs = 14
    if game_date is not None and home_team and away_team:
        ax.set_title(
            f"{stadium_name} 최근 {n}경기 실제 vs 예측 + "
            f"{pd.Timestamp(game_date).strftime('%Y-%m-%d')} {home_team} vs {away_team}",
            fontsize=title_fs,
            pad=14,
        )
    else:
        ax.set_title(
            f"{stadium_name} 최근 {n}경기 실제 vs 예측 + 이번 경기",
            fontsize=title_fs,
            pad=14,
        )
    apply_axes_theme(ax, th)
    add_chart_legend(
        ax,
        th,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=min(5, 2 + len(future_items)),
        fontsize=11,
    )
    plt.yticks(color=th.tick)
    fig.subplots_adjust(bottom=0.30, top=0.90, left=0.08, right=0.98)
    finalize_figure_for_streamlit(fig, th)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _aggregate_tree_importance_from_pipe(pipe) -> pd.Series:
    """OneHot+수치 파이프라인에서 원본 피처명 기준으로 중요도 합산 (RF/LGBM/XGB 공통)."""
    from modeling.train_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    reg = pipe.regressor_
    pre = reg.named_steps["preprocess"]
    rf = reg.named_steps["model"]
    s = pd.Series(rf.feature_importances_, index=pre.get_feature_names_out())
    out: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        k = f"num__{col}"
        if k in s.index:
            out[col] = float(s[k])
    for col in CATEGORICAL_FEATURES:
        p = f"cat__{col}_"
        m = s.index.astype(str).str.startswith(p)
        out[col] = float(s.loc[m].sum()) if m.any() else 0.0
    return pd.Series(out).sort_values(ascending=False)


@st.cache_data(show_spinner=False)
def _cached_tree_feature_importance_series(model_path_str: str, mtime_key: int) -> pd.Series:
    p = Path(model_path_str)
    if not p.exists():
        return pd.Series(dtype=float)
    pipe = joblib.load(p)
    return _aggregate_tree_importance_from_pipe(pipe)


def _importance_pct_grouped(imp: pd.Series) -> pd.Series:
    g = _group_rf_importance_for_display(imp)
    tot = float(g.sum()) or 1.0
    return (g / tot * 100.0).astype(float)


def _feature_keys_for_grouped_chart(imps_pct: dict[str, pd.Series], top_n: int = 12) -> list[str]:
    """모델 간 최대 기여(%) 기준 상위 피처부터 내림차순 (차트·표 공통)."""
    scores: dict[str, float] = {}
    for imp_pct in imps_pct.values():
        for k, v in imp_pct.items():
            scores[k] = max(scores.get(k, 0.0), float(v))
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_n]


def _plot_grouped_feature_importance(
    imps_pct: dict[str, pd.Series],
    *,
    top_n: int = 12,
) -> plt.Figure:
    """선택한 알고리즘별 피처 중요도(%) — 가로 grouped bar."""
    keys = _feature_keys_for_grouped_chart(imps_pct, top_n=top_n)
    labels = [_ko_ml_feature_label(str(k)) for k in keys]
    n_feat = len(keys)
    model_names = list(imps_pct.keys())
    n_models = len(model_names)
    y = np.arange(n_feat)
    group_h = 0.78
    bar_h = group_h / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.38 * n_feat)))
    th = apply_figure_theme(fig, ax)

    for i, name in enumerate(model_names):
        vals = [float(imps_pct[name].get(k, 0.0)) for k in keys]
        offset = (i - (n_models - 1) / 2) * bar_h
        ax.barh(
            y + offset,
            vals,
            bar_h * 0.92,
            label=name,
            color=model_bar_color(name),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=th.tick, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("상대 기여 (모델 내 %)", fontsize=11)
    ax.set_title(
        "알고리즘별 피처 중요도 비교 (날씨 세부는 2그룹 합산)",
        fontsize=13,
    )
    apply_axes_theme(ax, th)
    add_chart_legend(ax, th, loc="lower right", fontsize=9)
    fig.tight_layout()
    finalize_figure_for_streamlit(fig, th)
    return fig


def _load_importance_pct_for_model(model_label: str) -> pd.Series:
    imp_fname = next(
        (fname for _k, label, fname in ML_MODEL_REGISTRY if label == model_label),
        None,
    )
    if imp_fname is None:
        return pd.Series(dtype=float)
    imp_path = PROJECT_ROOT / "models" / imp_fname
    try:
        mt = int(os.path.getmtime(imp_path))
    except OSError:
        mt = 0
    imp = _cached_tree_feature_importance_series(str(imp_path), mt)
    if len(imp) == 0:
        return pd.Series(dtype=float)
    return _importance_pct_grouped(imp)


_ML_FEAT_LABEL_KO: dict[str, str] = {
    "연도": "연도",
    "월": "월",
    "주차_ISO": "주차(ISO)",
    "stadium_capacity": "구장 정원",
    "is_capacity_missing": "정원 결측 여부",
    "is_rain": "강수(0/1)",
    "is_hot": "폭염(30℃+)",
    "is_weekend": "주말 여부",
    "is_friday": "금요일",
    "is_saturday": "토요일",
    "is_sunday": "일요일",
    "weekday_sin": "요일 sin",
    "weekday_cos": "요일 cos",
    "is_small_stadium": "소형 구장 여부",
    "is_derby": "더비 매치",
    "is_season_opener": "시즌 초반(오프너)",
    "is_childrens_day": "어린이날·인접일",
    "home_win_rate": "홈팀 승률(시점)",
    "visitor_win_rate": "원정 승률(시점)",
    "win_rate_diff": "승률 차",
    "home_gb_to_5th": "홈 5위와 게임차",
    "visitor_gb_to_5th": "원정 5위와 게임차",
    "is_pennant_race": "페넌트레이스 구간",
    "playoff_urgency": "플레이오프 긴박도",
    "month_x_playoff_urgency": "월 × 긴박도",
    "home_prior_mean_att": "홈팀 과거 평균 관중",
    "visitor_prior_mean_att": "원정 과거 평균 관중",
    "home_last5_mean_att": "홈 최근 5경기 평균 관중",
    "visitor_last5_mean_att": "원정 최근 5경기 평균 관중",
    "home_draw_pct_in_league": "홈 리그 무승부 비율",
    "visitor_away_draw_pct_in_league": "원정 원정 무승부 비율",
    "home_visitor_prior_draw_diff": "무승부 비율 차",
    "matchup_prior_mean_att": "매치업 과거 평균 관중",
    "season_progress": "시즌 진행도",
    "홈팀": "홈팀(범주)",
    "방문팀": "원정팀(범주)",
    "구장": "구장(범주)",
    "rain_bucket": "강수 구간(범주)",
    "temp_bucket": "기온 구간(범주)",
    "humidity_bucket": "습도 구간(범주)",
    "wind_bucket": "풍속 구간(범주)",
    "stadium_x_rain": "구장×강수(범주)",
    "날씨(강수·구장연동)": "날씨(강수·구장연동)",
    "날씨(기온·습도·풍)": "날씨(기온·습도·풍)",
}

_ML_IMP_RAIN_GROUP = frozenset({"rain_bucket", "is_rain", "stadium_x_rain"})
_ML_IMP_THERMO_GROUP = frozenset({"temp_bucket", "humidity_bucket", "wind_bucket", "is_hot"})
_ML_IMP_WEATHER_DISPLAY_KEYS = ("날씨(강수·구장연동)", "날씨(기온·습도·풍)")

_TEMP_BUCKET_KO: dict[str, str] = {
    "VeryCold": "매우 쌀쌀",
    "Cold": "쌀쌀",
    "Mild": "쾌적",
    "Warm": "따뜻",
    "Hot": "더움",
}
_RAIN_BUCKET_KO: dict[str, str] = {
    "No_Rain": "무강수",
    "Rain_0_1mm": "미세 강수(~1mm)",
    "Rain_1_5mm": "소강수(1~5mm)",
    "Rain_5mm_plus": "강한 강수(5mm+)",
}
_HUM_BUCKET_KO: dict[str, str] = {
    "Dry": "건조",
    "Normal": "보통",
    "Humid": "다습",
    "VeryHumid": "매우 다습",
}
_WIND_BUCKET_KO: dict[str, str] = {
    "Calm": "잔풍",
    "Light": "약풍",
    "Moderate": "보통",
    "Strong": "강풍",
}


def _ko_ml_feature_label(name: str) -> str:
    return _ML_FEAT_LABEL_KO.get(str(name), str(name))


def _ml_cell_display(v: object) -> object:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
            return round(x, 4)
        return round(x, 6)
    return str(v)


def _group_rf_importance_for_display(imp: pd.Series) -> pd.Series:
    """날씨 세부 피처를 두 그룹으로 합산해 막대·표에서 한눈에 보이게 함."""
    s = imp.astype(float)
    rain_keys = [k for k in _ML_IMP_RAIN_GROUP if k in s.index]
    therm_keys = [k for k in _ML_IMP_THERMO_GROUP if k in s.index]
    g_r = float(s[rain_keys].sum()) if rain_keys else 0.0
    g_t = float(s[therm_keys].sum()) if therm_keys else 0.0
    rest = s.drop(labels=rain_keys + therm_keys, errors="ignore")
    extra = pd.Series(
        {"날씨(강수·구장연동)": g_r, "날씨(기온·습도·풍)": g_t},
        dtype=float,
    )
    out = pd.concat([rest, extra])
    return out.sort_values(ascending=False)


def _ml_prediction_snapshot(
    row: pd.Series,
    temp_c: float,
    rain_mm: float,
    hum_pct: float,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for c in (
        "홈팀",
        "방문팀",
        "구장",
        "연도",
        "월",
        "주차_ISO",
        "is_weekend",
        "stadium_capacity",
        "is_derby",
        "matchup_prior_mean_att",
        "home_prior_mean_att",
        "visitor_prior_mean_att",
        "home_win_rate",
        "visitor_win_rate",
    ):
        if c not in row.index:
            continue
        out[_ko_ml_feature_label(c)] = _ml_cell_display(row[c])

    rb = str(row.get("rain_bucket", "") or "").strip()
    tb = str(row.get("temp_bucket", "") or "").strip()
    hb = str(row.get("humidity_bucket", "") or "").strip()
    wb_raw = row.get("wind_bucket")
    if wb_raw is None or pd.isna(wb_raw):
        wb = ""
    else:
        wb = str(wb_raw).strip()
    out["강수(구간·입력)"] = f"{_RAIN_BUCKET_KO.get(rb, rb)} ({rain_mm:g}mm)"
    out["기온(구간·입력)"] = f"{_TEMP_BUCKET_KO.get(tb, tb)} ({temp_c:g}℃)"
    out["습도(구간·입력)"] = f"{_HUM_BUCKET_KO.get(hb, hb)} ({hum_pct:g}%)"
    out["풍속(구간)"] = _WIND_BUCKET_KO.get(wb, wb or "—")
    return out


def _plot_rf_importance_barh(
    imp: pd.Series, top_n: int = 15, *, model_label: str = "RandomForest"
) -> plt.Figure:
    """상위 top_n 피처를 기여(%) 내림차순으로 표시 (맨 위 = 가장 중요)."""
    s = imp.astype(float)
    tail = s.sort_values(ascending=False).head(top_n)
    labels = [_ko_ml_feature_label(str(i)) for i in tail.index]
    vals = tail.to_numpy(dtype=float)
    tot = float(vals.sum()) or 1.0
    pct = vals / tot * 100.0

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(tail))))
    th = apply_figure_theme(fig, ax)
    y = np.arange(len(tail))
    ax.barh(y, pct, color=model_bar_colors().random_forest, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=th.tick, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("상대 기여 (전체 중 %)", fontsize=11)
    ax.set_title(
        f"{model_label} 피처 중요도 (날씨 세부는 2그룹으로 합산)",
        fontsize=13,
    )
    apply_axes_theme(ax, th)
    fig.tight_layout()
    finalize_figure_for_streamlit(fig, th)
    return fig


# (key, 표시 이름, joblib 파일명) — benchmark_models.py 와 동일
ML_MODEL_REGISTRY: list[tuple[str, str, str]] = [
    ("rf", "RandomForest", "attendance_rf_pipeline.joblib"),
    ("lgbm", "LightGBM", "attendance_lgbm_pipeline.joblib"),
    ("xgb", "XGBoost", "attendance_xgb_pipeline.joblib"),
]


def render_ml_feature_importance_ui(
    model_labels: list[str],
    *,
    selectbox_key: str = "ml_feat_imp_model",
) -> None:
    """선택한 ML 모델의 피처 중요도 막대·표 (한 경기·CSV 일괄 공통)."""
    if not model_labels:
        return

    with st.expander("피처 중요도", expanded=False):
        imps_pct: dict[str, pd.Series] = {}
        for label in model_labels:
            pct = _load_importance_pct_for_model(label)
            if len(pct) > 0:
                imps_pct[label] = pct

        if not imps_pct:
            st.info("피처 중요도를 불러오지 못했습니다.")
            return

        if len(imps_pct) > 1:
            fig_imp = _plot_grouped_feature_importance(imps_pct, top_n=12)
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            keys = _feature_keys_for_grouped_chart(imps_pct, top_n=15)
            tbl_rows: list[dict[str, object]] = []
            for k in keys:
                row: dict[str, object] = {"피처": _ko_ml_feature_label(str(k))}
                for label in model_labels:
                    row[label] = round(float(imps_pct.get(label, pd.Series(dtype=float)).get(k, 0.0)), 2)
                tbl_rows.append(row)
            st.dataframe(pd.DataFrame(tbl_rows), width="stretch", hide_index=True)
        else:
            only_label = next(iter(imps_pct))
            imp_fname = next(
                (fname for _k, label, fname in ML_MODEL_REGISTRY if label == only_label),
                None,
            )
            imp_raw = pd.Series(dtype=float)
            if imp_fname is not None:
                imp_path = PROJECT_ROOT / "models" / imp_fname
                try:
                    mt = int(os.path.getmtime(imp_path))
                except OSError:
                    mt = 0
                imp_raw = _cached_tree_feature_importance_series(str(imp_path), mt)
            imp_disp = _group_rf_importance_for_display(imp_raw)
            fig_imp = _plot_rf_importance_barh(
                imp_disp, top_n=15, model_label=only_label
            )
            st.pyplot(fig_imp)
            plt.close(fig_imp)
            tbl = imp_disp.sort_values(ascending=False).head(20).reset_index()
            tbl.columns = ["피처", "기여(%)"]
            tbl["피처"] = tbl["피처"].map(lambda x: _ko_ml_feature_label(str(x)))
            st.dataframe(tbl, width="stretch", hide_index=True)


@st.cache_resource
def _load_ml_pipeline(model_filename: str):
    p = PROJECT_ROOT / "models" / model_filename
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource
def _load_rf_pipeline():
    return _load_ml_pipeline("attendance_rf_pipeline.joblib")


@st.cache_data
def _load_kbo_train_ready() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(ttl=600)
def _cached_forecast_rain_ref(stadium: str, game_start_iso: str, _auth_fp: str):
    """동네예보(typ02) 개시 3시간 전 참고. 키는 캐시용 지문만 사용."""
    from common import kma_vilage_fcst as kv

    auth = os.environ.get("KMA_APIHUB_AUTH_KEY", "").strip()
    if not auth:
        try:
            auth = str(st.secrets.get("KMA_APIHUB_AUTH_KEY", "")).strip()
        except Exception:
            pass
    gs = pd.Timestamp(game_start_iso)
    if gs.tzinfo is None:
        gs = gs.tz_localize(kv.KST)
    else:
        gs = gs.tz_convert(kv.KST)
    return kv.forecast_ref_for_rain_cancel_rules(stadium, gs, auth_key=auth or None)


# =========================
# CSS 스타일
# =========================
_ensure_session_data()
df = st.session_state.df

_app_css_path = Path(__file__).resolve().parent / "styles" / "app.css"
if _app_css_path.exists():
    st.markdown(
        f"<style>{_app_css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "앱 스타일 파일이 없습니다: `scripts/app/styles/app.css`",
        icon="⚠️",
    )

# =========================
# 사이드바
# =========================
st.sidebar.title("⚾ KBO Attendance Predictor")

_ml_train_path = PROJECT_ROOT / "data" / "processed" / "kbo_train_ready.csv"
_ml_train_ok = _ml_train_path.exists()


def _ml_model_available(fname: str) -> bool:
    return (PROJECT_ROOT / "models" / fname).is_file() and _ml_train_ok


st.sidebar.markdown("**ML 알고리즘**")
_ml_help = (
    "켜면 한 경기·CSV 일정 모두에 학습된 파이프라인을 적용합니다. "
    "여러 개를 켜면 **예측값은 평균**입니다. "
    "모델 파일은 `scripts/modeling/benchmark_models.py` 로 생성합니다."
)
use_ml_rf = st.sidebar.checkbox(
    "RandomForest",
    value=_ml_model_available("attendance_rf_pipeline.joblib"),
    disabled=not _ml_model_available("attendance_rf_pipeline.joblib"),
    help=_ml_help,
)
use_ml_lgbm = st.sidebar.checkbox(
    "LightGBM",
    value=False,
    disabled=not _ml_model_available("attendance_lgbm_pipeline.joblib"),
    help=_ml_help,
)
use_ml_xgb = st.sidebar.checkbox(
    "XGBoost",
    value=False,
    disabled=not _ml_model_available("attendance_xgb_pipeline.joblib"),
    help=_ml_help,
)
use_ml_models = use_ml_rf or use_ml_lgbm or use_ml_xgb
_ml_choice_map = {
    "rf": use_ml_rf,
    "lgbm": use_ml_lgbm,
    "xgb": use_ml_xgb,
}
if use_ml_models and not _ml_train_ok:
    st.sidebar.caption("ML: `kbo_train_ready.csv` 없음 → 휴리스틱만 사용됩니다.")
_missing = [
    label
    for key, label, fname in ML_MODEL_REGISTRY
    if _ml_choice_map.get(key) and not _ml_model_available(fname)
]
if _missing:
    st.sidebar.caption(f"파일 없음(학습 필요): {', '.join(_missing)}")

_ml_chosen_labels: list[str] = []
if use_ml_rf:
    _ml_chosen_labels.append("RandomForest")
if use_ml_lgbm:
    _ml_chosen_labels.append("LightGBM")
if use_ml_xgb:
    _ml_chosen_labels.append("XGBoost")

temperature = st.sidebar.slider("예상 기온(℃)", -10, 40, 23)
rainfall_mm = st.sidebar.slider(
    "일 합계 강수(mm)",
    0.0,
    120.0,
    0.0,
    0.5,
    help=(
        "**RandomForest** 입력의 `rain_bucket`·`is_rain` 등에 반영됩니다. "
        "CSV에 강수 열이 없으면 이 값을 모든 경기에 적용합니다."
    ),
)
humidity = st.sidebar.slider("예상 습도(%)", 0, 100, 60)
wind_speed = st.sidebar.slider(
    "예상 풍속(m/s)",
    0.0,
    15.0,
    2.0,
    0.1,
    help="ML 모델의 `wind_bucket`에 반영됩니다 (학습·추론 동일 구간).",
)

st.sidebar.markdown("---")
_MODE_SINGLE = "한 경기 예측"
_MODE_CSV = "CSV 업로드 예측"
input_mode = st.sidebar.radio(
    "예측 방식",
    [_MODE_SINGLE, _MODE_CSV],
    horizontal=True,
)
st.sidebar.markdown("---")

from modeling.batch_feature_builder import schedule_template_path
from app.csv_batch_predict_ui import (
    clear_batch_schedule_session,
    maybe_refresh_batch_predictions,
    on_sidebar_schedule_upload_cleared,
    render_csv_batch_results_main,
    try_auto_apply_schedule_csv,
    _batch_session_active,
)

_sched_tpl_path = schedule_template_path(PROJECT_ROOT)
if input_mode == _MODE_CSV:
    st.sidebar.markdown("**경기 일정 CSV**")
    if _sched_tpl_path.is_file():
        st.sidebar.download_button(
            "샘플 CSV",
            data=_sched_tpl_path.read_bytes(),
            file_name=_sched_tpl_path.name,
            mime="text/csv",
            use_container_width=True,
        )

if "sidebar_schedule_csv_nonce" not in st.session_state:
    st.session_state["sidebar_schedule_csv_nonce"] = 0
_uploader_key = f"sidebar_schedule_csv_{st.session_state['sidebar_schedule_csv_nonce']}"
_schedule_csv_upload = None
if input_mode == _MODE_CSV:
    _schedule_csv_upload = st.sidebar.file_uploader(
        "경기 일정 CSV",
        type=["csv"],
        key=_uploader_key,
        help="필수: 경기날짜, 홈팀, 방문팀, 구장 · 올리면 자동 예측",
    )

if _schedule_csv_upload is None:
    if _batch_session_active():
        on_sidebar_schedule_upload_cleared()
        st.session_state.pop("_csv_apply_error", None)
        st.rerun()
else:
    _csv_sig = f"{_schedule_csv_upload.name}:{_schedule_csv_upload.size}"
    from app.csv_batch_predict_ui import _ml_models_session_key as _batch_ml_models_key

    _ml_key = _batch_ml_models_key(_ml_chosen_labels)
    _csv_needs_run = (
        st.session_state.get("batch_upload_sig") != _csv_sig
        or "batch_attendance_result" not in st.session_state
        or st.session_state.get("batch_ml_models_key") != _ml_key
    )
    if _csv_needs_run:
        with st.spinner("일정 CSV 예측 중…"):
            _csv_err = try_auto_apply_schedule_csv(
                _schedule_csv_upload,
                _ml_chosen_labels,
                default_temp=float(temperature),
                default_rain=float(rainfall_mm),
                default_hum=float(humidity),
                cap_by_stadium=st.session_state.cap_by_stadium,
                ml_train_ok=_ml_train_ok,
            )
        if _csv_err:
            st.session_state["_csv_apply_error"] = _csv_err
            st.session_state["batch_upload_sig"] = _csv_sig
        else:
            st.session_state.pop("_csv_apply_error", None)

if st.session_state.get("_csv_apply_error"):
    st.sidebar.error(st.session_state["_csv_apply_error"])

if _batch_session_active():
    maybe_refresh_batch_predictions(
        _ml_chosen_labels,
        default_temp=float(temperature),
        default_rain=float(rainfall_mm),
        default_hum=float(humidity),
        cap_map=st.session_state.cap_by_stadium,
        ml_train_ok=_ml_train_ok,
    )

if input_mode == _MODE_CSV and _batch_session_active() and _schedule_csv_upload is not None:
    if st.sidebar.button(
        "CSV 제거·초기화",
        use_container_width=True,
        key="btn_clear_batch_schedule",
    ):
        # 파일 업로더 자체를 비우기 위해 key nonce 회전
        st.session_state["sidebar_schedule_csv_nonce"] = (
            int(st.session_state.get("sidebar_schedule_csv_nonce", 0)) + 1
        )
        clear_batch_schedule_session()
        st.rerun()

st.sidebar.markdown("---")
if input_mode == _MODE_SINGLE:
    st.sidebar.markdown("**한 경기 입력**")
    game_date = st.sidebar.date_input("경기 날짜", key="fld_game_date")
else:
    game_date = st.session_state.get("fld_game_date", pd.Timestamp.today().date())

_stadium_opts = sorted(df["구장"].dropna().unique())
_home_opts = sorted(df["홈팀"].dropna().unique())
_away_opts = sorted(df["방문팀"].dropna().unique())

if input_mode == _MODE_SINGLE:
    stadium = st.sidebar.selectbox(
        "경기장",
        _stadium_opts,
        key="fld_stadium",
        help="구장을 바꾸면 **홈팀**이 이 구장의 기본 홈(잠실→LG 등)으로 맞춰집니다.",
    )

    if "_prev_stadium_for_home" not in st.session_state:
        st.session_state._prev_stadium_for_home = None
    if st.session_state._prev_stadium_for_home != stadium:
        st.session_state._prev_stadium_for_home = stadium
        _dh = _default_home_team_for_stadium(stadium, df)
        if _dh in _home_opts:
            st.session_state.fld_home_team = _dh
        _cur_away = st.session_state.get("fld_away_team")
        if _cur_away is None or _cur_away == st.session_state.get("fld_home_team"):
            for _a in _away_opts:
                if _a != st.session_state.get("fld_home_team"):
                    st.session_state.fld_away_team = _a
                    break

    home_team = st.sidebar.selectbox(
        "홈팀",
        _home_opts,
        key="fld_home_team",
    )

    away_team = st.sidebar.selectbox(
        "원정팀",
        _away_opts,
        key="fld_away_team",
    )

    auto_recent_kbo = st.sidebar.checkbox(
        "최근 5경기 KBO 자동 반영",
        value=_default_web_recent_enabled(),
        help=(
            "선택한 경기 날짜 이전에 치른 직전 5경기를 GraphDaily에서 가져옵니다. "
            "Chrome·Selenium 필요(Streamlit Cloud에서는 기본 꺼짐). 꺼두면 로컬 CSV만 사용합니다. "
            f"캐시 TTL {_KBO_RECENT_TTL}초."
        ),
    )
else:
    stadium = str(st.session_state.get("fld_stadium") or _stadium_opts[0])
    home_team = str(st.session_state.get("fld_home_team") or _home_opts[0])
    away_team = str(st.session_state.get("fld_away_team") or _away_opts[0])
    auto_recent_kbo = _default_web_recent_enabled()

# =========================
# 메인 — 공통 제목·예측 방식
# =========================
_hdr_title, _hdr_btn = st.columns([7, 1], vertical_alignment="center")
with _hdr_title:
    st.markdown(
        '<div class="main-title">📈 KBO 관람 수요 예측 시스템</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-text">단일 경기 예측 또는 CSV 업로드 예측을 선택해 결과를 확인하세요.</div>',
        unsafe_allow_html=True,
    )
with _hdr_btn:
    if st.button(
        "🔄 새로고침",
        use_container_width=True,
        help="화면·차트·테마를 다시 불러옵니다.",
        key="btn_app_refresh",
    ):
        st.rerun()
st.markdown("---")

if input_mode == _MODE_CSV:
    if _batch_session_active():
        st.session_state["batch_feat_imp_renderer"] = render_ml_feature_importance_ui
        render_csv_batch_results_main(
            chosen=_ml_chosen_labels,
            cap_by_stadium=st.session_state.cap_by_stadium,
            ml_train_ok=_ml_train_ok,
            attendance_df=df,
            embedded=True,
            try_kbo_scrape=not _is_streamlit_cloud(),
        )
    else:
        st.info("사이드바에서 경기 일정 CSV를 업로드하면 예측 결과가 표시됩니다.")
    st.stop()

# =========================
# 예측: 휴리스틱 + (옵션) ML 파이프라인
# =========================
home_slice = df[
    (df["구장"] == stadium)
    & (df["홈팀"] == home_team)
]
visitor_slice = df[
    (df["구장"] == stadium)
    & (df["방문팀"] == away_team)
]
matchup_slice = df[
    (df["구장"] == stadium)
    & (df["홈팀"] == home_team)
    & (df["방문팀"] == away_team)
]

pred_source = ""

if len(matchup_slice) >= 1:

    pred_source = "matchup"
    predicted_heuristic = int(
        matchup_slice["관중수"].mean()
    )

elif len(home_slice) >= 1 and len(visitor_slice) >= 1:

    pred_source = "blend"
    predicted_heuristic = int(
        (
            home_slice["관중수"].mean()
            + visitor_slice["관중수"].mean()
        )
        / 2
    )

elif len(home_slice) >= 1:

    pred_source = "home_only"
    predicted_heuristic = int(
        home_slice["관중수"].mean()
    )

elif len(visitor_slice) >= 1:

    pred_source = "visitor_only"
    predicted_heuristic = int(
        visitor_slice["관중수"].mean()
    )

else:

    stadium_only = df[df["구장"] == stadium]

    if len(stadium_only) > 0:

        pred_source = "stadium_only"
        predicted_heuristic = int(
            stadium_only["관중수"].mean()
        )

    else:

        pred_source = "global"
        predicted_heuristic = int(
            df["관중수"].mean()
        )

# 휴리스틱: 강수 시 관중 약간 하향(RF의 rain_bucket과 별개; 일 강수 슬라이더 > 0일 때만).
if rainfall_mm > 0:
    predicted_heuristic = int(predicted_heuristic * 0.93)

if humidity >= 85:

    predicted_heuristic = int(
        predicted_heuristic * 0.98
    )

stadium_capacity = get_capacity(stadium, home_team)

predicted_heuristic = min(
    predicted_heuristic,
    max(1, int(stadium_capacity)),
)

ml_used = False
ml_row_snapshot: dict[str, object] | None = None
ml_predictions: dict[str, int] = {}
predicted_attendance = predicted_heuristic
_single_game_result_row: pd.DataFrame | None = None

if use_ml_models and _ml_train_ok:

    tr = _load_kbo_train_ready()

    if tr is not None:

        try:

            from modeling.batch_feature_builder import build_ml_feature_dataframe

            cap_map = st.session_state.get("cap_by_stadium") or _load_app_capacity_map()
            X = build_ml_feature_dataframe(
                tr,
                home=home_team,
                away=away_team,
                stadium=stadium,
                game_date=game_date,
                temp_c=float(temperature),
                rain_mm=float(rainfall_mm),
                hum_pct=float(humidity),
                cap_map=cap_map,
                wind_mps=float(wind_speed),
            )
            for col in X.columns:

                if X[col].dtype == object:

                    X[col] = X[col].astype(str).fillna("missing")

            cap_hi = max(1, int(stadium_capacity))
            for key, label, fname in ML_MODEL_REGISTRY:
                if not _ml_choice_map.get(key):
                    continue
                pipe = _load_ml_pipeline(fname)
                if pipe is None:
                    continue
                raw_ml = float(np.asarray(pipe.predict(X))[0])
                pred_i = int(round(max(0.0, raw_ml)))
                ml_predictions[label] = min(pred_i, cap_hi)

            if ml_predictions:
                predicted_attendance = int(round(float(np.mean(list(ml_predictions.values())))))
                ml_used = True
                ml_row_snapshot = _ml_prediction_snapshot(
                    X.iloc[0],
                    float(temperature),
                    float(rainfall_mm),
                    float(humidity),
                )

        except Exception as e:

            logger.warning("ML 파이프라인 예측 실패, 휴리스틱 유지: %s", e, exc_info=True)
            st.caption("모델 예측에 실패해 **과거 패턴 기반 추정(휴리스틱)** 값을 표시합니다.")

if ml_used:

    _ml_labels = ", ".join(ml_predictions.keys())
    if len(ml_predictions) > 1:
        st.caption(
            f"예측에 **{_ml_labels}** 모델을 반영했습니다. "
            f"큰 숫자는 **평균**, 차트 맨 오른쪽은 **알고리즘별** 막대입니다."
        )
    else:
        st.caption(f"예측에 **{_ml_labels}** 모델을 반영했습니다.")

    render_ml_feature_importance_ui(
        list(ml_predictions.keys()),
        selectbox_key="ml_feat_imp_model",
    )

    if ml_used and ml_row_snapshot:
        with st.expander("이번 입력 요약", expanded=False):
            _snap_rows = []
            for k, v in ml_row_snapshot.items():
                if isinstance(v, bool):
                    _disp = "예" if v else "아니오"
                else:
                    _disp = str(v)
                _snap_rows.append({"항목": k, "값": _disp})
            st.dataframe(
                pd.DataFrame(_snap_rows),
                width="stretch",
                hide_index=True,
            )

else:

    if pred_source in ("stadium_only", "global"):

        st.info(
            "선택한 팀·구장 조합의 과거 경기가 적어 "
            "**구장 또는 리그 전체 평균**으로 추정했습니다."
        )

    elif pred_source == "visitor_only":

        st.caption(
            "해당 구장에서 홈으로 치른 **홈팀** 기록이 없어, "
            "**원정팀 방문** 기록 위주로 추정했습니다."
        )

    elif pred_source == "home_only":

        st.caption(
            "해당 구장에서 **원정팀**이 온 기록이 없어, "
            "**홈팀 홈** 기록만으로 추정했습니다."
        )

    elif pred_source == "blend":

        st.caption(
            "동일 매치업 기록이 없어, **홈팀 홈 평균**과 "
            "**원정팀 방문 평균**을 반반 반영했습니다."
        )

    elif pred_source == "matchup":

        st.caption(
            "동일 **홈·원정·구장** 조합의 과거 관중 평균을 사용했습니다."
        )

    if use_ml_models and not ml_used:

        st.caption(
            "ML 알고리즘을 켰지만 필요한 모델·데이터 파일이 없어 **휴리스틱만** 사용했습니다. "
            "`benchmark_models.py` 실행 또는 체크박스 도움말(?)을 확인하세요."
        )

# CSV 일괄 예측과 합쳐 볼 수 있도록 현재 사이드바 입력도 1행으로 구성
try:
    _single_actual_val = _resolve_selected_game_actual(
        df,
        game_date,
        home_team,
        away_team,
        stadium,
        try_kbo=auto_recent_kbo and not _is_streamlit_cloud(),
    )
    _single_actual = float(_single_actual_val) if _single_actual_val is not None else np.nan
    _single_row: dict[str, object] = {
        "경기날짜": pd.Timestamp(game_date).strftime("%Y-%m-%d"),
        "홈팀": home_team,
        "방문팀": away_team,
        "구장": stadium,
        "관중수": _single_actual,
        "예측_관중수": int(predicted_attendance),
    }
    for _lbl, _val in ml_predictions.items():
        _single_row[f"예측_{_lbl}"] = int(_val)
    if len(ml_predictions) > 1:
        _single_row["예측_평균"] = int(predicted_attendance)
    _single_game_result_row = pd.DataFrame([_single_row])
except Exception:
    _single_game_result_row = None

# =========================
# 혼잡도 계산
# =========================
congestion = (
    predicted_attendance /
    stadium_capacity
) * 100

# =========================
# 운영 단계
# =========================
_plan = classify_congestion_pct(congestion)
level = _plan.level
action_class = _plan.action_class
action_title = _plan.action_title
action_msg = _plan.action_msg

st.markdown("## 📅 경기 정보")

st.markdown(
    f'<div class="match-text">'
    f'{html.escape(str(game_date))} | '
    f'{html.escape(str(home_team))} vs {html.escape(str(away_team))} | '
    f'{html.escape(str(stadium))}'
    f'</div>',
    unsafe_allow_html=True,
)

# =========================
# 예상 관중 수
# =========================
st.markdown(f"""
<div class="predict-box">

<div class="predict-label">
예상 관중 수
</div>

<div class="predict-number">
{predicted_attendance:,} 명
</div>

</div>
""", unsafe_allow_html=True)

_wx_debug_ui = os.environ.get("STREAMLIT_DEBUG_WEATHER", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

from common.kbo_regular_start_time import default_start_hm
from common import kma_vilage_fcst as kv

_h_f, _m_f = default_start_hm(game_date, game_date.year)
_gs_fc = pd.Timestamp(
    year=game_date.year,
    month=game_date.month,
    day=game_date.day,
    hour=_h_f,
    minute=_m_f,
    tz=kv.KST,
)
_t3 = kv.three_hours_before_game_start(_gs_fc)
_gs_s = html.escape(_gs_fc.strftime("%Y-%m-%d %H:%M"))
_t3_s = html.escape(_t3.strftime("%Y-%m-%d %H:%M"))
st.markdown(
    f"""
<div class="rain-info-compact">
<b>☂️ 우천·관중 안내</b> (관중 예측과 별개 · 참고용)
<ul>
<li>관중 수는 <b>정상 개최</b> 가정. 우천 취소·노게임은 반영하지 않음.</li>
<li>언론상 검토: 개시 <b>3시간 전</b> 강우 예보 · 개시 <b>1시간 전</b> 실제 강우 (실제와 다를 수 있음).</li>
<li>기상청 동네예보: 개시 3시간 전 <b>RN1</b>, 없으면 <b>POP</b> (API 키 필요).</li>
</ul>
개시 <b>{_gs_s}</b> (KST) · 참고 <b>{_t3_s}</b>
</div>
""",
    unsafe_allow_html=True,
)
_auth_fc = os.environ.get("KMA_APIHUB_AUTH_KEY", "").strip()
if not _auth_fc:
    try:
        _auth_fc = str(st.secrets.get("KMA_APIHUB_AUTH_KEY", "")).strip()
    except Exception:
        pass
_fp = hashlib.sha256(_auth_fc.encode()).hexdigest()[:16] if _auth_fc else "nokey"
_ref = _cached_forecast_rain_ref(stadium, _gs_fc.isoformat(), _fp)

if _wx_debug_ui:
    logger.info(
        "kma_fcst_ref stadium=%r game_start=%s ref3h=%s payload=%s",
        stadium,
        _gs_fc.isoformat(),
        str(_t3),
        {
            k: _ref.get(k)
            for k in (
                "ok",
                "mode",
                "base",
                "nx",
                "ny",
                "detail",
                "msg",
                "target_kst",
                "pop_anchor_kst",
            )
        },
    )
    with st.expander("기상·API 디버그 (`STREAMLIT_DEBUG_WEATHER=1`)", expanded=False):
        st.json(_ref)

if not _ref.get("ok"):
    _m = str(_ref.get("msg") or "예보 없음")
    _d = _ref.get("detail")
    _d_s = str(_d).strip() if _d is not None else ""
    _hint = html.escape(redact_api_secrets(_d_s)) if _d_s else ""
    st.markdown(
        f'<div class="rain-fcst-warn">'
        f'<div class="rain-fcst-warn-title">동네예보</div>'
        f'<p style="margin:0;">{html.escape(_m)}'
        + (f" <span class=\"rain-fcst-warn-tech\">({_hint})</span>" if _hint else "")
        + "</p></div>",
        unsafe_allow_html=True,
    )
else:
    _gout = kv.rainout_cancel_guidance(_ref)
    _gband = str(_gout["band"])
    _wrap_g = "rain-fcst-warn" if _gband == "warn" else f"rain-risk-box rain-risk-{_gband}"
    _head = html.escape(str(_gout["headline"]))
    _detail = ""
    if _ref.get("mode") == "ultra_rn1":
        mm = float(_ref["mm_h"])
        _detail = f"RN1 ≈ {mm:g} mm/h"
    elif _ref.get("mode") == "vilage_pop":
        _detail = f"POP ≈ {float(_ref['pop_pct']):.0f}%"
    _line = ""
    if _gout.get("lines"):
        _line = html.escape(
            redact_api_secrets(str(_gout["lines"][0]).replace("**", ""))
        )
    st.markdown(
        f'<div class="{_wrap_g}">'
        f'<div class="rain-risk-title">{_head}</div>'
        f'<p style="margin:6px 0 0 0;">'
        + (f"<b>{html.escape(_detail)}</b> · " if _detail else "")
        + (_line if _line else "비공식 참고 · 확정 아님")
        + "</p></div>",
        unsafe_allow_html=True,
    )

# =========================
# 정보 카드
# =========================
col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    👥 경기장 수용 인원
    </div>

    <div class="card-value">
    {stadium_capacity:,} 명
    </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    📊 혼잡도
    </div>

    <div class="card-value">
    {congestion:.1f}%
    </div>

    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    🛡 운영 강도
    </div>

    <div class="card-high">
    {level}
    </div>

    </div>
    """, unsafe_allow_html=True)

# =========================
# 액션 플랜
# =========================
st.markdown("## 🛡 상황별 액션 플랜")

st.markdown(f"""
<div class="{html.escape(action_class)}">

<h2>{html.escape(action_title)}</h2>

<p style="font-size:18px; line-height:1.7;">
{html.escape(action_msg)}
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# 최근 경기 그래프
# =========================
st.markdown(
    "## 📊 해당 경기장 최근 5경기 관중 수 추이"
)

_by_st = df[df["구장"] == stadium].dropna(subset=["경기날짜"])
_local = (
    _by_st[_by_st["경기날짜"] < pd.Timestamp(game_date)]
    .sort_values("경기날짜", ascending=False)
    .head(5)
    .sort_values("경기날짜")
)

recent_games = _local
chart_source = "로컬 CSV"

if auto_recent_kbo:
    scraped = _recent_five_kbo(stadium, game_date.isoformat())
    if scraped is not None and len(scraped) >= 1:
        recent_games = scraped
        chart_source = "KBO GraphDaily"
    else:
        st.caption(
            "KBO 자동 반영에 실패해 **로컬 CSV**로 표시합니다. "
            "(Chrome·네트워크 확인, 또는 옵션을 끄세요.)"
        )

st.caption(
    f"출처: **{chart_source}** · 기준일(그날 0시 **이전** 경기만): **{game_date}**"
)

if recent_games.empty:
    st.info(
        "이 구장·기준일 이전에 표시할 과거 경기가 없습니다. "
        "경기 날짜를 늦추거나 구장을 바꿔 보세요."
    )
else:
    hist = recent_games[["경기날짜", "홈팀", "방문팀", "관중수"]].copy()
    if "구장" in recent_games.columns:
        hist["구장"] = recent_games["구장"]
    hist["경기날짜"] = pd.to_datetime(hist["경기날짜"], errors="coerce")
    hist["관중수"] = pd.to_numeric(hist["관중수"], errors="coerce")

    compare_rows: list[dict] = []
    tr_chart = _load_kbo_train_ready() if (use_ml_models and _ml_train_ok) else None
    cap_map_chart = st.session_state.get("cap_by_stadium") or _load_app_capacity_map()

    if tr_chart is not None:
        for _, r in hist.iterrows():
            if pd.isna(r["경기날짜"]) or pd.isna(r["관중수"]):
                continue
            gdt = pd.Timestamp(r["경기날짜"])
            home_h = str(r["홈팀"]).strip()
            away_h = str(r["방문팀"]).strip()
            st_h = str(r.get("구장", stadium)).strip() or stadium
            actual_i = int(r["관중수"])
            temp, rain, hum, wind = _weather_for_historical_row(df, r)
            by_model = _predict_ml_attendance_by_model_for_row(
                tr_chart,
                home=home_h,
                away=away_h,
                stadium_name=st_h,
                game_date=gdt,
                temp_c=temp,
                rain_mm=rain,
                hum_pct=hum,
                wind_mps=wind,
                cap_map=cap_map_chart,
                ml_choice_map=_ml_choice_map,
            )
            pred_i = (
                int(round(float(np.mean(list(by_model.values())))))
                if by_model
                else np.nan
            )
            compare_rows.append(
                {
                    "경기": f"{gdt.strftime('%m/%d')}\n{home_h} vs {away_h}",
                    "실제": actual_i,
                    "예측": pred_i,
                    "예측_모델": by_model,
                    "오차": abs(actual_i - pred_i) if by_model else np.nan,
                }
            )

    if compare_rows:
        compare_df = pd.DataFrame(compare_rows)
        _sel_actual = _resolve_selected_game_actual(
            df,
            game_date,
            home_team,
            away_team,
            stadium,
            try_kbo=auto_recent_kbo and not _is_streamlit_cloud(),
        )
        _plot_recent_actual_vs_predicted(
            compare_df,
            future_pred=predicted_attendance,
            future_preds=ml_predictions if ml_predictions else None,
            stadium_name=stadium,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            selected_actual=_sel_actual,
        )
    else:
        st.warning(
            "ML 예측 비교를 만들지 못했습니다. 사이드바에서 **ML 알고리즘**을 켜고 "
            "`kbo_train_ready.csv`·joblib이 있는지 확인하세요."
        )
        chart_df = hist.copy()
        chart_df["경기정보"] = (
            chart_df["경기날짜"].dt.strftime("%m/%d")
            + "\n"
            + chart_df["홈팀"]
            + " vs "
            + chart_df["방문팀"]
        )
        chart_df.loc[len(chart_df)] = [
            pd.NaT,
            home_team,
            away_team,
            predicted_attendance,
            _selected_match_chart_label(game_date, home_team, away_team),
        ]
        fig, ax = plt.subplots(figsize=(11, 4))
        th = apply_figure_theme(fig, ax)
        bars = ax.bar(chart_df["경기정보"], chart_df["관중수"])
        _ymax = float(pd.to_numeric(chart_df["관중수"], errors="coerce").fillna(0).max()) or 1.0
        _label_dy = max(_ymax * 0.015, 200.0)
        for i, bar in enumerate(bars):
            bar.set_color(
                th.accent_cyan if i == len(bars) - 1 else actual_bar_color()
            )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + _label_dy,
                f"{int(h):,}",
                ha="center",
                color=th.fg,
                fontsize=10,
            )
        ax.set_title(f"{stadium} 최근 경기 관중 + 이번 경기 예측")
        ax.set_ylabel("관중 수")
        ax.tick_params(colors=th.tick)
        ax.yaxis.label.set_color(th.fg)
        ax.title.set_color(th.fg)
        for spine in ax.spines.values():
            spine.set_color(th.spine)
        plt.xticks(rotation=0, color=th.tick)
        plt.yticks(color=th.tick)
        finalize_figure_for_streamlit(fig, th)
        st.pyplot(fig)
        plt.close(fig)

