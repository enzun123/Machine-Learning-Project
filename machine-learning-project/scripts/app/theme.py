"""Streamlit 라이트/다크 모드에 맞춘 차트·막대 색상."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.theme_watcher import css_color_luminance, is_light_from_watcher, theme_signature_from_session


@dataclass(frozen=True)
class ChartTheme:
    bg: str
    fg: str
    muted: str
    tick: str
    legend_bg: str
    legend_edge: str
    spine: str
    accent_cyan: str


@dataclass(frozen=True)
class ModelBarColors:
    random_forest: str
    lightgbm: str
    xgboost: str
    actual: str
    pred_single: str
    default: str


# Streamlit 기본 다크/라이트 secondary 배경에 맞춤
_CHART_DARK = ChartTheme(
    bg="#262730",
    fg="#fafafa",
    muted="#a3a8b4",
    tick="#fafafa",
    legend_bg="#262730",
    legend_edge="#262730",
    spine="#464855",
    accent_cyan="#18e6ff",
)

_CHART_LIGHT = ChartTheme(
    bg="#f0f2f6",
    fg="#31333f",
    muted="#6b7280",
    tick="#31333f",
    legend_bg="#f0f2f6",
    legend_edge="#f0f2f6",
    spine="#c4c6cf",
    accent_cyan="#0891b2",
)

_BARS_DARK = ModelBarColors(
    random_forest="#c084fc",
    lightgbm="#4ade80",
    xgboost="#fb923c",
    actual="#4f8cff",
    pred_single="#fbbf24",
    default="#94a3b8",
)

_BARS_LIGHT = ModelBarColors(
    random_forest="#7c3aed",
    lightgbm="#15803d",
    xgboost="#c2410c",
    actual="#2563eb",
    pred_single="#b45309",
    default="#64748b",
)


def _theme_option(key: str) -> str | None:
    try:
        value = st.get_option(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    return None


def _hex_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return 0.5
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def is_light_theme() -> bool:
    watched = is_light_from_watcher()
    if watched is not None:
        return watched
    try:
        theme_type = st.context.theme.type
        if theme_type == "light":
            return True
        if theme_type == "dark":
            return False
    except Exception:
        pass
    try:
        base = st.context.theme.base
        if base == "light":
            return True
        if base == "dark":
            return False
    except Exception:
        pass
    bg = _theme_option("theme.backgroundColor")
    if bg and bg.startswith("#"):
        return _hex_luminance(bg) > 0.5
    return False


def chart_theme() -> ChartTheme:
    light = is_light_theme()
    preset = _CHART_LIGHT if light else _CHART_DARK

    sig = theme_signature_from_session()
    if sig:
        parts = sig.split("|")
        if len(parts) >= 2:
            bg, fg = parts[0].strip(), parts[1].strip()
            if bg.startswith("rgb") and fg.startswith("rgb"):
                return ChartTheme(
                    bg=bg,
                    fg=fg,
                    muted=preset.muted,
                    tick=fg,
                    legend_bg=bg,
                    legend_edge=bg,
                    spine=preset.spine,
                    accent_cyan=preset.accent_cyan,
                )

    bg = _theme_option("theme.secondaryBackgroundColor") or _theme_option(
        "theme.backgroundColor"
    )
    text = _theme_option("theme.textColor")
    if bg and text:
        return ChartTheme(
            bg=bg,
            fg=text,
            muted=preset.muted,
            tick=text,
            legend_bg=bg,
            legend_edge=bg,
            spine=preset.spine,
            accent_cyan=preset.accent_cyan,
        )
    return preset


def model_bar_colors() -> ModelBarColors:
    return _BARS_LIGHT if is_light_theme() else _BARS_DARK


def model_bar_color(label: str) -> str:
    colors = model_bar_colors()
    return {
        "RandomForest": colors.random_forest,
        "LightGBM": colors.lightgbm,
        "XGBoost": colors.xgboost,
    }.get(label, colors.default)


def actual_bar_color() -> str:
    return model_bar_colors().actual


def pred_single_bar_color() -> str:
    return model_bar_colors().pred_single


def apply_axes_theme(ax, th: ChartTheme, *, grid: bool = False) -> None:
    ax.set_facecolor(th.bg)
    ax.tick_params(axis="both", colors=th.muted, labelsize=10)
    ax.xaxis.label.set_color(th.muted)
    ax.yaxis.label.set_color(th.muted)
    if ax.title:
        ax.title.set_color(th.fg)
    if grid:
        ax.grid(axis="x", color=th.spine, alpha=0.4, linewidth=0.5)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(th.spine)
        spine.set_linewidth(0.6)


def add_chart_legend(ax, th: ChartTheme, **kwargs):
    leg = ax.legend(
        frameon=True,
        facecolor=th.legend_bg,
        edgecolor=th.legend_bg,
        labelcolor=th.fg,
        framealpha=1.0,
        **kwargs,
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_linewidth(0)
        frame.set_edgecolor(th.legend_bg)
    return leg


def apply_figure_theme(fig, ax) -> ChartTheme:
    th = chart_theme()
    _paint_figure(fig, ax, th)
    apply_axes_theme(ax, th)
    return th


def finalize_figure_for_streamlit(fig, th: ChartTheme | None = None) -> ChartTheme:
    if th is None:
        th = chart_theme()
    for ax in fig.axes:
        _paint_figure(fig, ax, th)
        apply_axes_theme(ax, th)
    if not fig.axes:
        fig.patch.set_facecolor(th.bg)
        fig.patch.set_alpha(1.0)
    return th


def _paint_figure(fig, ax, th: ChartTheme) -> None:
    fig.patch.set_facecolor(th.bg)
    fig.patch.set_alpha(1.0)
    ax.set_facecolor(th.bg)
    if hasattr(ax, "patch"):
        ax.patch.set_alpha(1.0)
