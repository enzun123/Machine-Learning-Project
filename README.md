# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**현재 브랜치:** `feat/build-features-refactor` — `build_features.py` 단계 분리·`PRIOR_ATTENDANCE_FORM_COLS` 정리

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest 기반 회귀 모델**로 KBO 경기의 관중 수를 예측합니다. 이 브랜치는 **`final_dataset.csv` → `kbo_train_ready.csv`** 피처 생성 로직을 모듈화·정리합니다.

**이 브랜치에서 추가·정리한 내용**
- `create_features_pro()`를 단계별 함수로 분리 (용량·기상·요일·순위·폼)
- `PRIOR_ATTENDANCE_FORM_COLS` — 시점 이전 관중·폼·매치업·시즌 진행도 컬럼 목록 명확화
- `MODEL_READY_COLUMNS` — 학습용 최종 컬럼 스키마 고정
- 누수 방지: `add_season_form_and_draw_proxy()`는 **해당 경기 이전** 정보만 사용

---

## 📁 저장소 구조

```
Machine-Learning-Project/
├── README.md
└── machine-learning-project/
    ├── data/
    │   ├── processed/
    │   │   ├── final_dataset.csv      # 입력
    │   │   └── kbo_train_ready.csv    # ★ 출력 (이 브랜치 핵심)
    │   └── external/
    │       ├── kbo_stadium_info.csv
    │       └── kbo_standings_daily.csv
    ├── models/
    └── scripts/
        ├── features/
        │   └── build_features.py      # ★ 피처 엔지니어링
        ├── common/
        │   └── stadium_aliases.py
        ├── preprocessing/
        ├── modeling/
        ├── data_collection/
        └── app/
            └── streamlit_app.py
```

---

## 🔄 데이터 파이프라인 (피처 구간)

```
[preprocess_attendance_weather.py] → final_dataset.csv
        ↓
[build_features.py]  create_features_pro()
        ↓
kbo_train_ready.csv  (MODEL_READY_COLUMNS)
        ↓
[train_model.py]     → attendance_rf_pipeline.joblib
```

**선행 데이터:** 원시·기상·순위 수집이 끝난 `final_dataset.csv`, `kbo_stadium_info.csv`, `kbo_standings_daily.csv`

---

## 🚀 실행 방법

### 사전 준비

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

### 피처 생성 (이 브랜치 핵심)

```bash
cd machine-learning-project

# 선행: 전처리까지 완료된 상태
python3 scripts/preprocessing/preprocess_attendance_weather.py

# 피처 생성
python3 scripts/features/build_features.py
```

**입력**
- `data/processed/final_dataset.csv` (필수 컬럼: `연도`, `월`, `주차_ISO`, `홈팀`, `방문팀`, `구장`, `요일`, `경기날짜`, 기상 4종, `관중수`)
- `data/external/kbo_stadium_info.csv` (수용 인원)
- `data/external/kbo_standings_daily.csv` (승률·순위, 없으면 승률 0.5 대체)

**출력**
- `data/processed/kbo_train_ready.csv`

### 학습·앱

```bash
python3 scripts/modeling/train_model.py
export KMA_APIHUB_AUTH_KEY="발급받은_키"
streamlit run scripts/app/streamlit_app.py
```

---

## 🧩 `create_features_pro()` 단계

| 순서 | 함수 | 내용 |
|------|------|------|
| 1 | `_prepare_main_frame` | 구장 별칭, 숫자형 변환 |
| 2 | `_add_stadium_capacity_and_clip` | `stadium_capacity`, 관중수 정원 105% 클리핑 |
| 3 | `_add_weather_and_buckets` | 강수·기온·습도·풍 버킷, `is_rain`, `is_hot` |
| 4 | `_add_weekday_derby_and_size` | 요일 sin/cos, 더비, 소규모 구장 |
| 5 | `_add_calendar_standings_and_playoff` | 시즌 오프너·어린이날, 순위·페넌트·플레이오프 긴급도 |
| 6 | `add_season_form_and_draw_proxy` | `PRIOR_ATTENDANCE_FORM_COLS` (누수 없음) |

---

## 📊 생성 피처 요약

| 구분 | 컬럼 예 |
|------|---------|
| 구장·용량 | `stadium_capacity`, `is_capacity_missing`, `is_small_stadium` |
| 기상 | `rain_bucket`, `temp_bucket`, `humidity_bucket`, `wind_bucket`, `stadium_x_rain` |
| 요일 | `is_weekend`, `is_friday`/`saturday`/`sunday`, `weekday_sin`/`cos` |
| 순위·시즌 | `home_win_rate`, `visitor_win_rate`, `home_gb_to_5th`, `is_pennant_race`, `playoff_urgency` |
| 이력·폼 (`PRIOR_ATTENDANCE_FORM_COLS`) | `home_prior_mean_att`, `home_last5_mean_att`, `matchup_prior_mean_att`, `season_progress` 등 |
| 이벤트 | `is_derby`, `is_season_opener`, `is_childrens_day` |
| 타겟 | `관중수` |

`train_model.py`의 `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES`는 위 스키마와 맞춰져 있습니다.

---

## ⚙️ 모델 정보

- **알고리즘**: RandomForest Regressor (`log1p` 타겟)
- **검증**: `연도`·`월`·`주차_ISO` 시간 순 홀드아웃

| 지표 | Dummy | 구장 평균 | RandomForest |
|------|-------|-----------|--------------|
| MAE | ~4,673 | ~3,827 | **~1,943** |
| R² | ~-0.03 | ~0.37 | **~0.73** |

---

## 🔑 API·키

| 용도 | 설정 |
|------|------|
| 배치 기상(typ01) | `weather_api.py`의 `AUTH_KEY` (로컬만, 커밋 금지) |
| Streamlit 동네예보(typ02) | `KMA_APIHUB_AUTH_KEY` |

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `feat/build-features-refactor` | **현재** — 피처 파이프라인 리팩터 |
| `feat/kbo-scraping-and-standings` | 관중·순위 수집 |
| `feat/weather-kma-refactor` | KMA env 통일 |
| `feat/pkg-layout-and-config` | pyproject·config·logging |
| `develop` | 기능 통합 |

---

## 🙋 문의

- 팀장: 허은준
- 연락처: enzun123@gmail.com
