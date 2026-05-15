# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**현재 브랜치:** `feat/weather-kma-refactor` — 기상청 API(typ01·typ02) 인증 통일, 동네예보·우천 참고 UI, RF 피처 중요도 정리

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest 기반 회귀 모델**로 KBO 경기의 관중 수를 예측합니다. 수집 → 전처리 → 피처 생성 → 학습 → 추론까지 자동화하며, **Streamlit**에서 관중 예측과 함께 **동네예보 기반 우천 참고**를 제공합니다.

**이 브랜치에서 추가·정리한 내용**
- typ01(일별 관측)·typ02(동네예보) 모두 **`KMA_APIHUB_AUTH_KEY`** 환경변수로 인증
- `weather_api.py` — 요청 간격 `WEATHER_API_REQUEST_SLEEP_SEC`, 키 없으면 관측 API 건너뜀
- `common/kma_vilage_fcst.py` — 개시 3시간 전 RN1/POP 조회, `forecast_ref_for_rain_cancel_rules()` 단계 분리, `rainout_cancel_guidance()` 우천 안내
- Streamlit — 동네예보·우천 취소 참고 UI, API 키 로그 마스킹, RF 피처 중요도 날씨 2그룹 합산, 휴리스틱 강수 반영 수정

---

## 📁 저장소 구조

```
Machine-Learning-Project/
├── README.md
├── .gitignore
└── machine-learning-project/
    ├── data/
    │   ├── raw/
    │   ├── interim/            # weather_api 병합 결과
    │   ├── processed/
    │   └── external/           # weather_cache.json 등
    ├── models/
    ├── reports/eda/
    └── scripts/
        ├── app/
        │   └── streamlit_app.py
        ├── common/
        │   ├── kma_vilage_fcst.py   # 동네예보 typ02 (핵심)
        │   ├── stadium_aliases.py
        │   └── kbo_regular_start_time.py
        ├── data_collection/
        │   └── weather_api.py       # typ01 관측 배치
        ├── preprocessing/
        ├── features/
        ├── modeling/
        └── eda/
```

---

## 🔄 데이터 파이프라인

```
[kbo_scraping.py]              →  data/raw/kbo_{연도}_attendance.csv
[kbo_standings_scrape.py]      →  data/external/kbo_standings_daily.csv
[kbo_size.py]                  →  data/external/kbo_stadium_info.csv
        ↓
[weather_api.py]               →  data/interim/kbo_*_attendance_weather.csv  ← typ01, KMA_APIHUB_AUTH_KEY
        ↓
[preprocess_attendance_weather.py] → data/processed/final_dataset.csv
        ↓
[build_features.py]            →  data/processed/kbo_train_ready.csv
        ↓
[train_model.py]               →  models/attendance_rf_pipeline.joblib
        ↓
[streamlit_app.py]             →  예측 + 동네예보(typ02) 우천 참고
```

> 학습용 기상은 **과거 일별 관측(typ01)** 이고, 앱의 동네예보는 **실시간 참고(typ02)** 이며 관중 예측 수치에 직접 넣지 않습니다.

---

## 🚀 실행 방법

### 사전 준비

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
# 선택: export WEATHER_API_REQUEST_SLEEP_SEC=0.3
```

- Python **3.10+**
- 크롤링·최근 5경기: **Chrome + Selenium**
- typ02는 API허브에서 **동네예보·초단기/단기예보** 활용 신청 필요

### 권장 실행 순서

```bash
cd machine-learning-project

python3 scripts/data_collection/kbo_scraping.py
python3 scripts/data_collection/kbo_standings_scrape.py
python3 scripts/data_collection/kbo_size.py
python3 scripts/data_collection/weather_api.py

python3 scripts/preprocessing/preprocess_attendance_weather.py
python3 scripts/features/build_features.py
python3 scripts/modeling/train_model.py
python3 scripts/modeling/evaluate_model.py        # 선택
```

### Streamlit 웹앱

```bash
cd machine-learning-project
export KMA_APIHUB_AUTH_KEY="발급받은_키"
streamlit run scripts/app/streamlit_app.py
```

**실행 전 권장**
- `models/attendance_rf_pipeline.joblib`, `data/processed/kbo_train_ready.csv`
- `data/interim/kbo_2025_attendance_weather.csv` 등

| 환경변수 | 설명 |
|----------|------|
| `KMA_APIHUB_AUTH_KEY` | typ01·typ02 공통 (또는 `.streamlit/secrets.toml`) |
| `WEATHER_API_REQUEST_SLEEP_SEC` | `weather_api.py` 요청 간격(초, 기본 `0.3`) |
| `STREAMLIT_WEB_RECENT=0` | KBO 최근 5경기 자동 수집 끄기 |
| `STREAMLIT_DEBUG_WEATHER=1` | 동네예보 API 디버그 패널 |

---

## 🖥️ Streamlit UI 기능

| 기능 | 설명 |
|------|------|
| 관중 예측 | RandomForest 또는 과거 CSV 평균(휴리스틱) |
| 기상 입력 | 기온·**일 합계 강수(mm)**·습도 — RF `rain_bucket` 등에 반영 |
| 피처 중요도 | RF 사용 시 막대 그래프(날씨 세부는 2그룹 합산) |
| 동네예보 | 개시 **3시간 전** 초단기 RN1 → 없으면 단기 POP |
| 우천 참고 | `rainout_cancel_guidance` — KBO 운영 관행 대비 **참고용** (예측과 분리) |
| API 안전 | 오류 메시지·URL의 `authKey` 마스킹 |

---

## 📊 주요 스크립트

| 스크립트 | 역할 |
|----------|------|
| `data_collection/weather_api.py` | typ01 일별 관측 → interim (`KMA_APIHUB_AUTH_KEY`) |
| `common/kma_vilage_fcst.py` | typ02 RN1/POP, `forecast_ref_for_rain_cancel_rules`, 우천 안내 |
| `common/kbo_regular_start_time.py` | 구장·날짜별 가정 개시 시각 |
| `common/stadium_aliases.py` | 구장 이름 별칭 |
| `app/streamlit_app.py` | 예측·동네예보·우천 UI |
| `features/build_features.py` | 학습용 피처 |
| `modeling/train_model.py` | RF 파이프라인 학습 |
| 기타 `data_collection/*`, `preprocessing/*`, `modeling/*`, `eda/*` | 크롤링·전처리·평가·EDA |

---

## ⚙️ 모델 정보

- **알고리즘**: RandomForest Regressor (`Pipeline` + `OneHotEncoder`)
- **타겟**: `log1p` 변환
- **검증**: `연도`·`월`·`주차_ISO` 시간 순 홀드아웃
- **산출물**: `models/attendance_rf_pipeline.joblib`, `train_report.json`

| 지표 | Dummy | 구장 평균 | RandomForest |
|------|-------|-----------|--------------|
| MAE | ~4,673 | ~3,827 | **~1,943** |
| R² | ~-0.03 | ~0.37 | **~0.73** |

---

## 🔑 기상 API 설정

**하나의 키로 typ01·typ02를 모두 설정합니다.**

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
```

| 용도 | API | 모듈 |
|------|-----|------|
| 배치 학습용 일별 기온·강수·풍·습도 | typ01 관측 | `weather_api.py` |
| 앱 우천 참고(초단기 RN1·단기 POP) | typ02 동네예보 | `kma_vilage_fcst.py`, Streamlit |

- `WEATHER_API_REQUEST_SLEEP_SEC`: typ01 연속 요청 간격(기본 `0.3`초)
- 키가 비어 있으면 `weather_api.py`는 관측 요청을 **건너뜁니다**
- API 키는 **저장소에 커밋하지 마세요** (`.env`는 `.gitignore` 대상)

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `feat/weather-kma-refactor` | **현재** — KMA env 통일·동네예보·우천 UI |
| `feat/pkg-layout-and-config` | pyproject·config·logging |
| `develop` | 기능 통합 |
| `main` | 배포용 안정 버전 |

---

## 🙋 문의

- 팀장: 허은준
- 연락처: enzun123@gmail.com
