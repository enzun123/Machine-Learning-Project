"""
feat/eda: `final_dataset.csv` 기준 탐색 전용.

- 목적: 피처 설계 아이디어·리포트 생성. `kbo_train_ready` / 모델 입력 컬럼은 여기서 수정하지 않음.
- 구장 별칭: `common.stadium_aliases.STADIUM_ALIAS` (전처리·build_features와 동일 정본).
- 산출물: `reports/eda/figures/*.png` 및 `reports/eda/eda_summary.md`. 스크립트를 실행할 때마다
  같은 경로에 덮어쓰기로 갱신한다. 요약·그림을 Git에 올릴지는 팀에서 선택한다.
- 실행: `machine-learning-project` 디렉터리에서 `python3 scripts/eda/run_eda.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[2]
_scripts_dir = Path(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from common.stadium_aliases import STADIUM_ALIAS

DATA_PATH = ROOT_DIR / "data" / "processed" / "final_dataset.csv"
STADIUM_INFO_PATH = ROOT_DIR / "data" / "external" / "kbo_stadium_info.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "eda"
FIG_DIR = OUTPUT_DIR / "figures"
SUMMARY_PATH = OUTPUT_DIR / "eda_summary.md"
WEEKDAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    stadium_df = pd.read_csv(STADIUM_INFO_PATH, encoding="utf-8-sig")

    needed = [
        "관중수",
        "요일",
        "홈팀",
        "구장",
        "일평균기온(°C)",
        "일합계강수량(mm)",
        "일평균풍속(m/s)",
        "일평균상대습도(%)",
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"final_dataset.csv 필수 컬럼 누락: {miss}")

    if not {"구장", "최대수용인원"}.issubset(stadium_df.columns):
        raise KeyError("kbo_stadium_info.csv에 `구장`, `최대수용인원` 컬럼이 필요합니다.")

    for col in ["관중수", "일평균기온(°C)", "일합계강수량(mm)", "일평균풍속(m/s)", "일평균상대습도(%)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    stadium_df["최대수용인원"] = pd.to_numeric(stadium_df["최대수용인원"], errors="coerce")

    df = df.dropna(subset=["관중수"]).copy()
    stadium_df = stadium_df.dropna(subset=["구장", "최대수용인원"]).copy()

    # 구장 명칭 alias를 통합해 동일 구장으로 계산합니다.
    df["구장"] = df["구장"].replace(STADIUM_ALIAS)
    stadium_df["구장"] = stadium_df["구장"].replace(STADIUM_ALIAS)
    stadium_df = stadium_df.drop_duplicates("구장")
    return df, stadium_df


def plot_target_distribution(df: pd.DataFrame) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["관중수"], bins=30, kde=True, ax=ax, color="#2E86C1")
    ax.set_title("관중 수 분포 (Histogram + KDE)")
    ax.set_xlabel("관중 수")
    ax.set_ylabel("경기 수")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_target_attendance_distribution.png", dpi=150)
    plt.close(fig)

    return {
        "mean": float(df["관중수"].mean()),
        "median": float(df["관중수"].median()),
        "std": float(df["관중수"].std()),
        "q1": float(df["관중수"].quantile(0.25)),
        "q3": float(df["관중수"].quantile(0.75)),
        "p95": float(df["관중수"].quantile(0.95)),
    }


def plot_categorical(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    weekday_mean = df.groupby("요일", observed=True)["관중수"].mean().reindex(WEEKDAY_ORDER).dropna()
    team_mean = df.groupby("홈팀", observed=True)["관중수"].mean().sort_values(ascending=False)
    stadium_mean = df.groupby("구장", observed=True)["관중수"].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.barplot(x=weekday_mean.index, y=weekday_mean.values, ax=axes[0], color="#17A589")
    axes[0].set_title("요일별 평균 관중 수")
    axes[0].set_xlabel("요일")
    axes[0].set_ylabel("평균 관중 수")

    sns.barplot(x=team_mean.values, y=team_mean.index, ax=axes[1], color="#AF7AC5")
    axes[1].set_title("홈팀별 평균 관중 수")
    axes[1].set_xlabel("평균 관중 수")
    axes[1].set_ylabel("홈팀")

    sns.barplot(x=stadium_mean.values, y=stadium_mean.index, ax=axes[2], color="#F5B041")
    axes[2].set_title("구장별 평균 관중 수")
    axes[2].set_xlabel("평균 관중 수")
    axes[2].set_ylabel("구장")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_categorical_barplots.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.boxplot(data=df, x="요일", y="관중수", order=WEEKDAY_ORDER, ax=axes[0], color="#48C9B0")
    axes[0].set_title("요일별 관중 수")

    sns.boxplot(data=df, x="홈팀", y="관중수", order=team_mean.index, ax=axes[1], color="#BB8FCE")
    axes[1].set_title("홈팀별 관중 수")
    axes[1].tick_params(axis="x", rotation=45)

    sns.boxplot(data=df, x="구장", y="관중수", order=stadium_mean.index, ax=axes[2], color="#F8C471")
    axes[2].set_title("구장별 관중 수")
    axes[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_categorical_boxplots.png", dpi=150)
    plt.close(fig)

    return weekday_mean, team_mean, stadium_mean


def plot_capacity(df: pd.DataFrame, stadium_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(stadium_df[["구장", "최대수용인원"]], on="구장", how="left").dropna(subset=["최대수용인원"])
    merged["수용률"] = (merged["관중수"] / merged["최대수용인원"]).clip(lower=0)
    summary = merged.groupby("구장", observed=True)[["관중수", "최대수용인원", "수용률"]].mean().sort_values("수용률", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=summary.index, y=summary["수용률"] * 100, ax=ax, color="#E67E22")
    ax.set_title("구장별 평균 수용률(평균 관중 수 / 최대수용인원)")
    ax.set_xlabel("구장")
    ax.set_ylabel("평균 수용률(%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_stadium_capacity_ratio.png", dpi=150)
    plt.close(fig)
    return summary


def plot_weather(df: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    cols = ["일평균기온(°C)", "일합계강수량(mm)", "일평균풍속(m/s)", "일평균상대습도(%)", "관중수"]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("관중 수 및 기상 변수 피어슨 상관계수")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_weather_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    tmp = df.copy()
    tmp["비여부"] = tmp["일합계강수량(mm)"].apply(lambda x: "비(>0mm)" if pd.to_numeric(x, errors="coerce") > 0 else "무강수(0mm)")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=tmp, x="비여부", y="관중수", order=["무강수(0mm)", "비(>0mm)"], ax=ax, color="#3498DB")
    ax.set_title("강수 여부별 관중 수 비교")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_rainy_vs_non_rainy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=tmp, x="일평균기온(°C)", y="관중수", alpha=0.7, ax=ax, color="#9B59B6")
    sns.regplot(data=tmp, x="일평균기온(°C)", y="관중수", scatter=False, ci=None, ax=ax, color="black")
    ax.set_title("기온과 관중 수 산점도")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_temperature_scatter.png", dpi=150)
    plt.close(fig)

    weather_detail = {
        "rainy_mean": float(tmp.loc[tmp["비여부"] == "비(>0mm)", "관중수"].mean()),
        "non_rainy_mean": float(tmp.loc[tmp["비여부"] == "무강수(0mm)", "관중수"].mean()),
        "hot_mean": float(tmp.loc[tmp["일평균기온(°C)"] >= 30, "관중수"].mean()),
        "normal_mean": float(tmp.loc[tmp["일평균기온(°C)"] < 30, "관중수"].mean()),
    }
    weather_detail["rainy_drop"] = weather_detail["non_rainy_mean"] - weather_detail["rainy_mean"]
    return corr["관중수"].drop(index="관중수").sort_values(key=lambda s: s.abs(), ascending=False), weather_detail


def write_summary(
    target_stats: dict[str, float],
    weekday_mean: pd.Series,
    team_mean: pd.Series,
    capacity_summary: pd.DataFrame,
    corr_to_target: pd.Series,
    weather_detail: dict[str, float],
) -> None:
    hot_diff = weather_detail["hot_mean"] - weather_detail["normal_mean"]
    hot_trend = "증가" if hot_diff >= 0 else "감소"
    strongest_var = corr_to_target.index[0]
    strongest_corr = corr_to_target.iloc[0]

    lines = [
        "# EDA 결과 요약",
        "",
        "## 1) 타겟 변수(관중 수) 분포",
        f"- 평균: {target_stats['mean']:.0f}명 / 중앙값: {target_stats['median']:.0f}명 / 표준편차: {target_stats['std']:.0f}",
        f"- 사분위수(Q1~Q3): {target_stats['q1']:.0f}명 ~ {target_stats['q3']:.0f}명, 상위 95%: {target_stats['p95']:.0f}명",
        "",
        "## 2) 범주형 변수 분석",
        f"- 요일별 평균: 최고 `{weekday_mean.idxmax()}`({weekday_mean.max():.0f}명), 최저 `{weekday_mean.idxmin()}`({weekday_mean.min():.0f}명)",
        f"- 홈팀별 평균: 최고 `{team_mean.index[0]}`({team_mean.iloc[0]:.0f}명), 최저 `{team_mean.index[-1]}`({team_mean.iloc[-1]:.0f}명)",
        f"- 구장 수용률: 최고 `{capacity_summary.index[0]}`({capacity_summary.iloc[0]['수용률']*100:.1f}%), 최저 `{capacity_summary.index[-1]}`({capacity_summary.iloc[-1]['수용률']*100:.1f}%)",
        "",
        "## 3) 기상 데이터 상관관계",
        f"- 절대 상관이 가장 큰 변수: `{strongest_var}` (r={strongest_corr:.3f})",
        "",
        "## 4) 특정 기상 조건 비교",
        f"- 무강수 평균: {weather_detail['non_rainy_mean']:.0f}명 / 비(>0mm) 평균: {weather_detail['rainy_mean']:.0f}명",
        f"- 비가 오면 평균 약 {weather_detail['rainy_drop']:.0f}명 감소",
        f"- 30°C 이상 구간은 30°C 미만 대비 평균 약 {abs(hot_diff):.0f}명 {hot_trend}",
        "",
        "## 5) 피처 엔지니어링 제안",
        "- `강수 여부(0/1)` 이진 변수",
        "- `요일` 인코딩(원-핫/순환)",
        "- `수용률(관중수/최대수용인원)` 파생 변수",
        "- `폭염 여부(30°C+)`, `고습 구간` 파생 변수",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    df, stadium_df = load_data()

    target_stats = plot_target_distribution(df)
    weekday_mean, team_mean, _ = plot_categorical(df)
    capacity_summary = plot_capacity(df, stadium_df)
    corr_to_target, weather_detail = plot_weather(df)
    write_summary(target_stats, weekday_mean, team_mean, capacity_summary, corr_to_target, weather_detail)

    print(f"EDA 완료: {OUTPUT_DIR}")
    print(f"요약 파일: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
