"""
파이프라인 전역 상수 (마법 숫자 집약).

피처·전처리에서 공통으로 쓰는 임계값만 둔다. 도메인별 세부값은 각 모듈 docstring에 남긴다.
"""

from __future__ import annotations

# --- 시즌·구장 ---
FULL_SEASON_GAMES: int = 144
SMALL_STADIUM_CAPACITY: float = 15_000.0
ATTENDANCE_CAP_CLIP_MULTIPLIER: float = 1.05

# --- 강수 버킷 (일합계강수량 mm) ---
RAIN_BINS: tuple[float, ...] = (-1.0, 0.0, 1.0, 5.0, float("inf"))
RAIN_LABELS: tuple[str, ...] = ("No_Rain", "Rain_0_1mm", "Rain_1_5mm", "Rain_5mm_plus")

# --- 기상 결측·이상치 기본 ---
DEFAULT_TEMP_MEDIAN_FALLBACK: float = 15.0
DEFAULT_RH_MEDIAN_FALLBACK: float = 60.0
DEFAULT_WIND_MEDIAN_FALLBACK: float = 2.0
HOT_TEMP_THRESHOLD_C: float = 30.0

TEMP_BUCKET_EDGES: tuple[float, ...] = (-float("inf"), 10.0, 20.0, 25.0, 30.0, float("inf"))
TEMP_BUCKET_LABELS: tuple[str, ...] = ("VeryCold", "Cold", "Mild", "Warm", "Hot")

HUMIDITY_BUCKET_EDGES: tuple[float, ...] = (0.0, 40.0, 60.0, 80.0, 100.0)
HUMIDITY_BUCKET_LABELS: tuple[str, ...] = ("Dry", "Normal", "Humid", "VeryHumid")

WIND_BUCKET_EDGES: tuple[float, ...] = (-1.0, 1.5, 3.3, 5.4, float("inf"))
WIND_BUCKET_LABELS: tuple[str, ...] = ("Calm", "Light", "Moderate", "Strong")

# --- 순위·페넌트 ---
DEFAULT_WIN_RATE: float = 0.5
STANDINGS_DEFAULT_RANK_FOR_GB: int = 5
# 일별 순위표에서 5위 행 인덱스(0-based)
STANDINGS_FIFTH_PLACE_ROW_INDEX: int = 4
PENNANT_RACE_START_MONTH: int = 8
PENNANT_RACE_GB_ABS_MAX: float = 3.0

# --- 폼·플레이오프 보조 ---
FORM_LAST_N_GAMES: int = 5
PLAYOFF_URGENCY_BASE: float = 3.0
PLAYOFF_URGENCY_CLIP_HIGH: float = 3.0
PLAYOFF_URGENCY_CLIP_LOW: float = -10.0
