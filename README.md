# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**현재 브랜치:** `feat/kbo-scraping-and-standings` — KBO 관중·일자별 순위 스크랩 보강, 구장 `region_key` 단일화

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest 기반 회귀 모델**로 KBO 경기의 관중 수를 예측합니다. 이 브랜치는 **데이터 수집·구장 마스터**를 정비해, 이후 전처리·피처·학습 파이프라인이 같은 구장·지역 키를 쓰도록 맞춥니다.

**이 브랜치에서 추가·정리한 내용**
- `kbo_scraping.py` — GraphDaily 관중 수집, 날짜·시간 파싱 보강, `기상_매핑_지역키` 컬럼 추가
- `kbo_standings_scrape.py` — 기록실 일자별 팀 순위(정규시즌), 기간·raw CSV 기준일 지정 CLI
- `kbo_size.py` — `kbo_stadium_info.csv`에 **최대수용인원 + `region_key`** 생성
- `common/stadium_region.py` — 구장 → `region_key` 조회의 **단일 소스**(CSV 기준)
- `common/stadium_aliases.py` — 구장 표기 별칭 정본(전처리·피처·EDA와 동일)

---

## 📁 저장소 구조

```
Machine-Learning-Project/
├── README.md
├── .gitignore
└── machine-learning-project/
    ├── data/
    │   ├── raw/                    # kbo_{연도}_attendance.csv
    │   ├── interim/
    │   ├── processed/
    │   └── external/
    │       ├── kbo_stadium_info.csv    # 구장·수용인원·region_key
    │       └── kbo_standings_daily.csv # 일자별 순위 스냅샷
    ├── models/
    ├── reports/eda/
    └── scripts/
        ├── common/
        │   ├── stadium_region.py       # region_key 조회
        │   ├── stadium_aliases.py      # 구장 별칭
        │   ├── kbo_regular_start_time.py
        │   └── kma_vilage_fcst.py      # Streamlit 동네예보(typ02)
        ├── data_collection/
        │   ├── kbo_scraping.py         # ★ 관중 스크랩
        │   ├── kbo_standings_scrape.py # ★ 일별 순위
        │   ├── kbo_size.py             # ★ 구장 마스터
        │   ├── weather_api.py
        │   └── fetch_recent_crowd.py
        ├── preprocessing/
        ├── features/
        ├── modeling/
        ├── eda/
        └── app/
            └── streamlit_app.py
```

---

## 🔄 데이터 파이프라인

```
[kbo_scraping.py]              →  data/raw/kbo_{연도}_attendance.csv
                                  (기상_매핑_지역키 포함)
[kbo_standings_scrape.py]      →  data/external/kbo_standings_daily.csv
[kbo_size.py]                  →  data/external/kbo_stadium_info.csv
        ↓
[weather_api.py]               →  data/interim/kbo_*_attendance_weather.csv
        ↓
[preprocess_attendance_weather.py] → data/processed/final_dataset.csv
        ↓
[build_features.py]            →  data/processed/kbo_train_ready.csv
        ↓
[train_model.py]               →  models/attendance_rf_pipeline.joblib
        ↓
[streamlit_app.py]             →  웹 UI
```

**`region_key` 흐름:** `kbo_size.py` → `kbo_stadium_info.csv` → `stadium_region.stadium_to_region_key()` → raw의 `기상_매핑_지역키` 및 기상 API 구장 매핑

---

## 🚀 실행 방법

### 사전 준비

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

- Python **3.10+**
- 관중 스크랩: **Chrome + Selenium**
- 순위 스크랩: `requests`만 사용(기본)

### 1) 이 브랜치 핵심 — 데이터 수집

```bash
cd machine-learning-project

# 구장 마스터 (수용인원·region_key) — 먼저 실행 권장
python3 scripts/data_collection/kbo_size.py

# 관중 (GraphDaily, 기본 2024·2025)
python3 scripts/data_collection/kbo_scraping.py
python3 scripts/data_collection/kbo_scraping.py --years 2024 --headed   # 디버그 시

# 일자별 팀 순위 (정규시즌)
python3 scripts/data_collection/kbo_standings_scrape.py \
  --season-year 2024 --date-from 2024-03-23 --date-to 2024-10-01

# raw 관중 CSV에 있는 경기일만 순위 수집
python3 scripts/data_collection/kbo_standings_scrape.py \
  --season-year 2024 \
  --dates-from-raw data/raw/kbo_2024_attendance.csv data/raw/kbo_2025_attendance.csv \
  --skip-existing
```

| 옵션 / 변수 | 설명 |
|-------------|------|
| `kbo_scraping --years` | 수집 연도 (기본 `2024 2025`) |
| `kbo_scraping --headed` | 헤드리스 끄고 브라우저 표시 |
| `KBO_SCRAPE_HEADLESS=0` | 환경변수로 헤드리스 끄기 |
| `kbo_standings --skip-existing` | 이미 수집한 날짜 건너뜀 |
| `kbo_standings --sleep` | 요청 간 대기(초, 기본 `0.8`) |

**raw CSV 주요 컬럼:** `연도`, `경기날짜`, `경기시간`, `홈팀`, `방문팀`, `구장`, `기상_매핑_지역키`, `관중수`, `월`, `주차_ISO` 등

### 2) 이후 파이프라인

```bash
python3 scripts/data_collection/weather_api.py
python3 scripts/preprocessing/preprocess_attendance_weather.py
python3 scripts/features/build_features.py
python3 scripts/modeling/train_model.py
```

### Streamlit

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"   # 동네예보(typ02) 우천 참고
streamlit run scripts/app/streamlit_app.py
```

---

## 📊 주요 스크립트 (이 브랜치)

| 스크립트 | 역할 |
|----------|------|
| `data_collection/kbo_scraping.py` | GraphDaily 일별 관중·경기 메타, `region_key` 부여 |
| `data_collection/kbo_standings_scrape.py` | 기록실 일자별 팀 순위·승률 스냅샷 |
| `data_collection/kbo_size.py` | `kbo_stadium_info.csv` (구단·구장·수용인원·`region_key`) |
| `common/stadium_region.py` | 구장 문자열 → `region_key` (CSV 단일 소스) |
| `common/stadium_aliases.py` | 구장 표기 통일 (`STADIUM_ALIAS`) |
| `common/kbo_regular_start_time.py` | 표에 시간 없을 때 기본 개시 시각 |
| `preprocessing/preprocess_attendance_weather.py` | interim 병합 (별칭 적용) |
| `features/build_features.py` | 승률·매치업 등 피처 (`standings_daily` 조인) |

---

## ⚙️ 모델 정보

- **알고리즘**: RandomForest Regressor
- **검증**: `연도`·`월`·`주차_ISO` 시간 순 홀드아웃
- **산출물**: `models/attendance_rf_pipeline.joblib`

| 지표 | Dummy | 구장 평균 | RandomForest |
|------|-------|-----------|--------------|
| MAE | ~4,673 | ~3,827 | **~1,943** |
| R² | ~-0.03 | ~0.37 | **~0.73** |

---

## 🔑 API·키 설정

| 용도 | 설정 |
|------|------|
| 배치 기상(typ01) | `weather_api.py` 상단 `AUTH_KEY` (로컬만 수정, 커밋 금지) |
| Streamlit 동네예보(typ02) | `KMA_APIHUB_AUTH_KEY` 또는 `.streamlit/secrets.toml` |

구장·지역 매핑은 **`kbo_stadium_info.csv`의 `region_key`** 를 기준으로 유지합니다. 수용 인원·구장명 변경 시 `kbo_size.py`의 `STADIUM_ROWS`를 수정한 뒤 스크립트를 다시 실행하세요.

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `feat/kbo-scraping-and-standings` | **현재** — 관중·순위 스크랩, `region_key` |
| `feat/weather-kma-refactor` | KMA env 통일·동네예보 |
| `feat/pkg-layout-and-config` | pyproject·config·logging |
| `develop` | 기능 통합 |
| `main` | 배포용 안정 버전 |

---

## 🙋 문의

- 팀장: 허은준
- 연락처: enzun123@gmail.com
