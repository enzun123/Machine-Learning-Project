# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**현재 브랜치:** `feat/pkg-layout-and-config` — 패키지 설치(`pyproject.toml`)·공통 `config`·`logging`·기상 API(typ02) 권한 정리

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest 기반 회귀 모델**로 KBO 경기의 관중 수를 예측합니다. 수집 → 전처리 → 피처 생성 → 학습 → 추론까지 전 과정을 자동화하며, **Streamlit UI**에서 관중 예측·혼잡도·동네예보(우천 참고)를 확인할 수 있습니다.

**이 브랜치에서 추가·정리한 내용**
- `pyproject.toml` + `pip install -e .` 로 `scripts/` 하위를 패키지(`common`, `app`, `modeling` 등)로 설치
- `common/config.py` — 강수·기온 버킷, 시즌·순위 등 파이프라인 전역 상수 일원화
- `common/logging_config.py` — 배치 스크립트 공통 로깅 (`LOG_LEVEL` 환경변수)
- `common/kma_vilage_fcst.py` — 동네예보(typ02) 인증을 `KMA_APIHUB_AUTH_KEY` 환경변수로 통일

---

## 📁 저장소 구조

```
Machine-Learning-Project/
├── README.md
├── .gitignore
└── machine-learning-project/
    ├── pyproject.toml          # kbo-ml-scripts 패키지 정의 (필수 설치)
    ├── data/
    │   ├── raw/
    │   ├── interim/
    │   ├── processed/
    │   └── external/
    ├── models/
    ├── reports/eda/
    └── scripts/                # setuptools 패키지 루트 (where = ["scripts"])
        ├── app/
        │   └── streamlit_app.py
        ├── common/
        │   ├── config.py           # 전역 상수
        │   ├── logging_config.py   # setup_logging()
        │   ├── stadium_aliases.py
        │   ├── kma_vilage_fcst.py  # 동네예보 typ02
        │   └── kbo_regular_start_time.py
        ├── data_collection/
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
[weather_api.py]               →  data/interim/kbo_*_attendance_weather.csv
        ↓
[preprocess_attendance_weather.py] → data/processed/final_dataset.csv
        ↓
[build_features.py]            →  data/processed/kbo_train_ready.csv
        ↓
[train_model.py]               →  models/attendance_rf_pipeline.joblib
        ↓
[evaluate_model.py]            →  models/eval_report.json
        ↓
[streamlit_app.py]             →  🌐 웹 UI
```

---

## 🚀 실행 방법

### 사전 준비

```bash
cd machine-learning-project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

> `from common.xxx` import를 쓰는 모든 스크립트는 **`pip install -e .` 이후**에 실행하세요.  
> `scripts/` 디렉터리만 `cd` 해서 실행해도 되지만, editable 설치가 권장됩니다.

- Python **3.10+**
- 크롤링·최근 경기 자동 반영: **Chrome + Selenium**
- 동네예보(typ02): **`KMA_APIHUB_AUTH_KEY`** (Streamlit·`kma_vilage_fcst.py`)
- 배치 기상(typ01): `weather_api.py` 상단 `AUTH_KEY` (로컬 수정, 커밋 금지)

### 권장 실행 순서

```bash
cd machine-learning-project

python3 scripts/data_collection/kbo_scraping.py
python3 scripts/data_collection/kbo_standings_scrape.py
python3 scripts/data_collection/kbo_size.py
python3 scripts/data_collection/weather_api.py

python3 scripts/preprocessing/preprocess_attendance_weather.py
python3 scripts/features/build_features.py

python3 scripts/eda/run_eda.py                    # 선택
python3 scripts/modeling/train_model.py
python3 scripts/modeling/evaluate_model.py        # 선택
python3 scripts/modeling/tune_hyperparams.py --n-trials 50   # 선택
```

> 💡 `data/raw/`, `data/interim/` CSV가 있으면 해당 단계부터 시작할 수 있습니다.

### Streamlit 웹앱

```bash
cd machine-learning-project
export KMA_APIHUB_AUTH_KEY="발급받은_키"   # 우천 참고(typ02)용
streamlit run scripts/app/streamlit_app.py
```

**실행 전 권장**
- `models/attendance_rf_pipeline.joblib`
- `data/processed/kbo_train_ready.csv`
- `data/interim/kbo_2025_attendance_weather.csv` 등 (앱이 여러 경로에서 탐색)

| 환경변수 | 설명 |
|----------|------|
| `KMA_APIHUB_AUTH_KEY` | 동네예보 typ02 (또는 `.streamlit/secrets.toml`) |
| `LOG_LEVEL` | 배치 스크립트 로그 레벨 (기본 `INFO`) |
| `STREAMLIT_WEB_RECENT=0` | KBO 최근 5경기 자동 수집 끄기 |
| `STREAMLIT_DEBUG_WEATHER=1` | 동네예보 API 디버그 패널 |

---

## 🖥️ Streamlit UI 기능

| 기능 | 설명 |
|------|------|
| 사이드바 | 경기 날짜, 구장·홈·원정, 기온·강수·습도 |
| 예측 | RandomForest 또는 과거 CSV 평균(휴리스틱) |
| 혼잡도 | 수용률(%)·운영 단계 안내 |
| 최근 5경기 | 옵션 시 GraphDaily 크롤링 (Selenium) |
| 동네예보 | 개시 3시간 전 RN1/POP·우천 참고 (관중 예측과 분리) |

---

## 📊 주요 스크립트

| 스크립트 | 역할 |
|----------|------|
| `data_collection/kbo_scraping.py` | 일별 관중·경기 메타 크롤링 |
| `data_collection/kbo_standings_scrape.py` | 일자별 순위·승률 |
| `data_collection/kbo_size.py` | 구장 수용 인원 CSV |
| `data_collection/weather_api.py` | 기상청 typ01 관측 → interim |
| `data_collection/fetch_recent_crowd.py` | 구장별 최근 N경기 관중 |
| `preprocessing/preprocess_attendance_weather.py` | `final_dataset.csv` |
| `features/build_features.py` | `kbo_train_ready.csv` (`common.config` 사용) |
| `modeling/train_model.py` | RF 파이프라인 학습 |
| `modeling/evaluate_model.py` | 테스트 구간 평가 |
| `modeling/predict.py` | 배치 예측 |
| `modeling/tune_hyperparams.py` | Optuna 튜닝 |
| `eda/run_eda.py` | EDA 리포트 |
| `app/streamlit_app.py` | 관람 수요 예측 웹앱 |
| `common/config.py` | 버킷·시즌·순위 등 전역 상수 |
| `common/logging_config.py` | `setup_logging()` |
| `common/kma_vilage_fcst.py` | 동네예보 typ02 |
| `common/stadium_aliases.py` | 구장 이름 별칭 |

---

## ⚙️ 모델 정보

- **알고리즘**: RandomForest Regressor (`Pipeline` + `OneHotEncoder`)
- **타겟**: `log1p` 변환 (`TransformedTargetRegressor`)
- **검증**: `연도`·`월`·`주차_ISO` 시간 순 홀드아웃
- **산출물**: `models/attendance_rf_pipeline.joblib`, `train_report.json`, `eval_report.json`

**저장 리포트 기준 (테스트 구간, 참고)**

| 지표 | Dummy | 구장 평균 | RandomForest |
|------|-------|-----------|--------------|
| MAE | ~4,673 | ~3,827 | **~1,943** |
| R² | ~-0.03 | ~0.37 | **~0.73** |

---

## 🔑 기상 API 설정

| 용도 | API | 설정 위치 |
|------|-----|-----------|
| 일별 관측(배치) | typ01 | `weather_api.py`의 `AUTH_KEY` (로컬만 수정) |
| 동네예보(앱 참고) | typ02 | `export KMA_APIHUB_AUTH_KEY=...` 또는 Streamlit secrets |

- typ02는 API허브에서 **동네예보·초단기/단기예보** 활용 신청 필요
- API 키·`.env`는 **저장소에 커밋하지 마세요**

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `feat/pkg-layout-and-config` | **현재** — pyproject·config·logging·typ02 인증 |
| `develop` | 기능 통합 (Streamlit UX·피처 등 병합 대상) |
| `main` | 배포용 안정 버전 |
| `feat/*` | 기능별 개발 |

---

## 🙋 문의

- 팀장: 허은준
- 연락처: enzun123@gmail.com
