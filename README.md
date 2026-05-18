# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**🌐 데모 (Streamlit Cloud):** [kbo-ml-prediction.streamlit.app](https://kbo-ml-prediction.streamlit.app)

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest 기반 회귀 모델**로 KBO 경기의 관중 수를 예측합니다. 수집 → 전처리 → 피처 생성 → 학습 → 추론까지 전 과정을 자동화하며, **Streamlit UI**에서 관중 예측·혼잡도·운영 액션 플랜·동네예보(우천 참고)를 한 화면에서 확인할 수 있습니다.

**주요 특징 (`main` / `develop` 동기화 기준)**
- `scripts/common/` 공통 모듈(설정·로깅·구장 별칭·동네예보·혼잡도 등)로 파이프라인·앱 로직 분리
- 피처: 승률·페넌트·플레이오프 긴급도·매치업·최근 5경기 평균·시즌 진행률 등
- 웹앱: RF 예측 + 휴리스틱 폴백, 피처 중요도, KBO 최근 5경기 자동 수집(로컬), 기상청 typ02 우천 참고
- Streamlit Cloud: 루트 `requirements.txt`·`packages.txt`로 배포, 한글 폰트·Chromium 드라이버 경로 대응

### 📦 저장소에 포함된 데이터 (학습·앱 생략 가능)

아래가 이미 커밋되어 있으면 **수집·학습 없이** Streamlit만 실행해도 됩니다.

| 경로 | 설명 |
|------|------|
| `data/raw/kbo_*_attendance.csv` | 연도별 관중 원시 |
| `data/interim/kbo_*_attendance_weather.csv` | 관중 + 기상 |
| `data/processed/kbo_train_ready.csv` | 학습용 피처 테이블 |
| `models/attendance_rf_pipeline.joblib` | 학습된 RF 파이프라인 |
| `models/train_report.json`, `eval_report.json` | 학습·평가 리포트 |

---

## 📁 저장소 구조

```
Machine-Learning-Project/
├── README.md
├── .gitignore
├── requirements.txt              # Streamlit Cloud / 루트 pip (-e ./machine-learning-project)
├── packages.txt                  # Cloud 시스템 패키지 (chromium, fonts-nanum 등)
└── machine-learning-project/
    ├── pyproject.toml          # 의존성·패키지 설치 (pip install -e .)
    ├── data/
    │   ├── raw/                # 연도별 일별 관중 원시 데이터
    │   ├── interim/            # 관중 + 기상 병합 중간 데이터
    │   ├── processed/          # final_dataset, kbo_train_ready (학습용 최종 테이블)
    │   └── external/           # 구장 정원·region_key, 일별 순위, 기상 캐시 등
    ├── models/                 # 학습 파이프라인(.joblib), 리포트, 튜닝 결과
    ├── reports/eda/            # EDA 산출물 (figures, eda_summary.md)
    └── scripts/
        ├── app/
        │   ├── streamlit_app.py
        │   ├── assets/fonts/            # Nanum Gothic (Cloud 한글 차트)
        │   └── styles/app.css           # 웹앱 커스텀 스타일
        ├── common/
        │   ├── config.py                # 파이프라인 전역 상수
        │   ├── logging_config.py
        │   ├── stadium_aliases.py
        │   ├── stadium_region.py        # 구장 → region_key
        │   ├── congestion_levels.py     # 혼잡도별 운영 액션
        │   ├── kma_vilage_fcst.py       # 동네예보(typ02) RN1/POP
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
[streamlit_app.py]             →  🌐 웹 UI (예측·혼잡도·동네예보 참고)
```

---

## 🚀 실행 방법

### 사전 준비

**권장: editable 설치**

```bash
cd machine-learning-project
pip install -e .
```

또는 개별 설치:

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

- Python **3.10+** (로컬 권장 **3.11~3.12** / Streamlit Cloud는 **3.12** 권장)
- 크롤링·최근 경기 자동 반영: **Chrome + Selenium** (로컬)
- 기상 API: 환경변수 **`KMA_APIHUB_AUTH_KEY`** (아래 [기상 API 설정](#-기상-api-설정) 참고)

**루트에서 Cloud와 동일하게 설치하려면**

```bash
pip install -r requirements.txt
```

### 권장 실행 순서

```bash
cd machine-learning-project

# 1) 원시 데이터 수집 (Selenium + Chrome 필요)
python3 scripts/data_collection/kbo_scraping.py
python3 scripts/data_collection/kbo_standings_scrape.py

# 2) 구장 수용 인원 CSV 생성
python3 scripts/data_collection/kbo_size.py

# 3) 기상 데이터 병합 → interim (KMA_APIHUB_AUTH_KEY 필요)
export KMA_APIHUB_AUTH_KEY="your_key_here"
python3 scripts/data_collection/weather_api.py

# 4) interim → final_dataset
python3 scripts/preprocessing/preprocess_attendance_weather.py

# 5) 피처 생성 → kbo_train_ready
python3 scripts/features/build_features.py

# 6) (선택) EDA 리포트 생성
python3 scripts/eda/run_eda.py

# 7) 모델 학습
python3 scripts/modeling/train_model.py

# 8) (선택) 평가 / 하이퍼파라미터 튜닝
python3 scripts/modeling/evaluate_model.py
python3 scripts/modeling/tune_hyperparams.py --n-trials 50
```

> 💡 위 [저장소에 포함된 데이터](#-저장소에-포함된-데이터-학습앱-생략-가능)가 있으면 해당 단계부터 시작하거나 앱만 실행하면 됩니다.

### Streamlit 웹앱 실행 (로컬)

```bash
cd machine-learning-project
streamlit run scripts/app/streamlit_app.py
```

**실행 전 권장 산출물**
- `models/attendance_rf_pipeline.joblib` (`train_model.py`)
- `data/processed/kbo_train_ready.csv`
- 관중·기상 CSV: `data/interim/kbo_2025_attendance_weather.csv` 등 (앱이 여러 경로에서 자동 탐색)

**선택 환경변수**

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `KMA_APIHUB_AUTH_KEY` | 동네예보(typ02) 우천 참고·`weather_api.py` 관측(typ01) | (없음) |
| `STREAMLIT_WEB_RECENT` | `0`이면 KBO GraphDaily 최근 5경기 자동 수집 **끔** | 로컬 **켜짐**, Cloud **꺼짐** |
| `STREAMLIT_DEBUG_WEATHER` | `1`이면 동네예보 API 디버그 패널 | 꺼짐 |

- 로컬 secrets: `machine-learning-project/.streamlit/secrets.toml`에 `KMA_APIHUB_AUTH_KEY` 설정 가능.
- Cloud secrets: 앱 설정 → Secrets에 동일 키 추가.

---

## ☁️ Streamlit Cloud 배포

| 설정 항목 | 값 |
|-----------|-----|
| Repository | `enzun123/Machine-Learning-Project` |
| Branch | `main` (또는 `develop` — 내용 동기화 시 동일) |
| Main file path | `machine-learning-project/scripts/app/streamlit_app.py` |
| Python | **3.12** 권장 |

**루트 파일 (자동 인식)**

- `requirements.txt` — `pip install -e ./machine-learning-project` 및 의존성
- `packages.txt` — `fonts-nanum`, `chromium`, `chromium-driver` (한글·Selenium)

**Secrets (권장)**

```toml
KMA_APIHUB_AUTH_KEY = "발급받은_키"
# Cloud에서 KBO 자동 크롤이 불안정하면 아래 추가
STREAMLIT_WEB_RECENT = "0"
```

**로컬 vs Cloud 차이**

| 기능 | 로컬 | Cloud |
|------|------|-------|
| RF 예측·혼잡도·피처 중요도 | ✅ | ✅ |
| 동네예보(typ02) | Secrets 키 필요 | 동일 |
| KBO 최근 5경기 자동 수집 | 기본 **ON** (Chrome) | 기본 ON 시 system chromium 사용 코드 포함 **OFF** (Selenium 제한)|
| 한글 차트 | OS 폰트 / Nanum 다운로드 | `packages.txt` + 앱 내 Nanum 등록 |

배포 후 **Manage app → Reboot**으로 반영합니다.

---

## 🖥️ Streamlit UI 기능

| 기능 | 설명 |
|------|------|
| 사이드바 입력 | 경기 날짜, 구장·홈·원정팀, 기온·일 강수(mm)·습도 |
| 구장 연동 | 구장 변경 시 기본 홈팀 자동 설정 |
| 예측 모드 | **RandomForest** 또는 **과거 CSV 평균 + 날씨 룰(휴리스틱)** |
| 피처 중요도 | RF 사용 시 막대 그래프·이번 입력 피처 요약 |
| 혼잡도·운영 | 수용률(%) → LOW / NORMAL / HIGH 및 매장·안전 액션 안내 |
| 최근 5경기 | 옵션 시 KBO 기록실에서 구장별 최근 경기 차트 (Selenium) |
| 동네예보 참고 | 개시 3시간 전 RN1/POP·우천 취소 참고 (관중 예측값과 분리) |
| 스타일 | `scripts/app/styles/app.css` 커스텀 UI |

---

## 📊 주요 스크립트

| 스크립트 | 역할 |
|----------|------|
| `data_collection/kbo_scraping.py` | 일별 관중·경기 메타 크롤링 |
| `data_collection/kbo_standings_scrape.py` | 일자별 팀 순위·승률 스냅샷 |
| `data_collection/kbo_size.py` | 구단·구장별 최대 수용 인원 CSV |
| `data_collection/weather_api.py` | 기상청 API허브 typ01 관측 → interim |
| `data_collection/fetch_recent_crowd.py` | 구장별 최근 N경기 관중 (CLI·앱 연동) |
| `preprocessing/preprocess_attendance_weather.py` | interim 병합 → `final_dataset.csv` |
| `features/build_features.py` | 학습용 피처(승률·페넌트·매치업·폼 등) |
| `modeling/train_model.py` | 시간 순 홀드아웃 + RF 파이프라인 학습 |
| `modeling/evaluate_model.py` | 테스트 구간 재평가·리포트 |
| `modeling/predict.py` | 저장 모델 배치 예측 |
| `modeling/tune_hyperparams.py` | Optuna + TimeSeriesSplit 튜닝 |
| `eda/run_eda.py` | EDA 리포트·차트 생성 |
| `app/streamlit_app.py` | 관람 수요 예측·혼잡도·기상 참고 웹앱 |
| `common/config.py` | 강수·기온 버킷, 시즌·순위 임계값 등 |
| `common/logging_config.py` | 파이프라인·앱 공통 로깅 |
| `common/stadium_aliases.py` | 구장·팀 표기 정규화 |
| `common/stadium_region.py` | 구장 → 기상 region_key |
| `common/kbo_regular_start_time.py` | 정규시즌 기본 개시 시각 |
| `common/kma_vilage_fcst.py` | 동네예보 typ02 (RN1/POP) |
| `common/congestion_levels.py` | 혼잡도 구간별 운영 메시지 |

---

## ⚙️ 모델 정보

- **알고리즘**: RandomForest Regressor (`sklearn` Pipeline + `OneHotEncoder`)
- **타겟 변환**: `log1p` (`TransformedTargetRegressor`)
- **검증**: `연도`·`월`·`주차_ISO` 기준 시간 순 홀드아웃 (테스트 약 20%)
- **저장**: `models/attendance_rf_pipeline.joblib`, `models/test_indices.npy`, `models/train_report.json`
- **튜닝(선택)**: Optuna + TimeSeriesSplit → `models/best_params.json`
- **선택 모델**: LightGBM (`train_model.py`에서 `HAS_LGBM`일 때만)

**현재 저장 리포트 기준 (테스트 구간, 참고용)**

| 지표 | Dummy(전역 평균) | 구장 평균 | RandomForest |
|------|------------------|-----------|--------------|
| MAE | ~4,673 | ~3,827 | **~1,943** |
| R² | ~-0.03 | ~0.37 | **~0.73** |

주요 피처 예: `matchup_prior_mean_att`, `home_prior_mean_att`, `stadium_capacity`, `home_last5_mean_att`, 요일·승률·페넌트·강수/기온 버킷 등 (`build_features.py`·`train_model.py` 참고).

---

## 🔑 기상 API 설정

기상청 **API허브** 인증키는 코드에 하드코딩하지 않고 환경변수로 받습니다.

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
```

| 용도 | API | 사용 위치 |
|------|-----|-----------|
| 일별 관측(기온·강수 등) | typ01 | `weather_api.py` |
| 동네예보(초단기 RN1·단기 POP) | typ02 | `kma_vilage_fcst.py`, Streamlit |

- typ02는 API허브에서 **동네예보·초단기/단기예보** 활용 신청이 필요합니다.
- 요청 간격: `WEATHER_API_REQUEST_SLEEP_SEC` (기본 `0.3`초)
- `.env`는 `.gitignore`에 포함되어 있습니다. **키를 저장소에 커밋하지 마세요.**

---

## 🌿 Git 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | **배포·Streamlit Cloud** 기본 브랜치 (`develop` 병합 반영) |
| `develop` | 기능 통합 브랜치 (Streamlit UX·피처·Cloud 패치) |
| `feat/*` / `feature/*` | 기능별 개발 브랜치 (히스토리 보존) |

`main`과 `develop`은 Cloud·chromedriver·한글 폰트 등 배포 수정이 맞춰진 상태를 유지합니다.

---

## 👥 팀 · 문의

| 역할 | 이름 | 비고 |
|------|------|------|
| 팀장 | 허은준 (enzun123) | enzun123@gmail.com |
| 팀원 | Kimjiwo_243, leesm9840, whddnjs448-hash |

- **문의:** enzun123@gmail.com
- **용도:** 교육·팀 프로젝트 (KBO 관중 예측 ML).
