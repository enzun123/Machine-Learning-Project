# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest 기반 회귀 모델**로 KBO 경기의 관중 수를 예측합니다. 수집 → 전처리 → 피처 생성 → 학습 → 추론까지 전 과정을 자동화하며, **Streamlit UI**를 통해 손쉽게 인터랙티브하게 예측 결과를 확인할 수 있습니다.

---

## 📁 저장소 구조

```
Machine-Learning-Project/
├── README.md
├── .gitignore
└── machine-learning-project/
    ├── data/
    │   ├── raw/               # 연도별 일별 관중 원시 데이터
    │   ├── interim/           # 관중 + 기상 병합 중간 데이터
    │   ├── processed/         # final_dataset, kbo_train_ready (학습용 최종 테이블)
    │   └── external/          # 구장 정원, 일별 순위, 기상 캐시 등
    ├── models/                # 학습된 파이프라인(.joblib), 리포트, 튜닝 결과
    ├── reports/eda/           # EDA 산출물 (figures, eda_summary.md)
    └── scripts/
        ├── app/
        │   └── streamlit_app.py          # 웹 UI
        ├── common/
        │   └── stadium_aliases.py        # 구장 이름 별칭 공통 모듈
        ├── data_collection/              # 크롤링 · 기상 · 최근 N경기 수집
        ├── preprocessing/               # 전처리 병합
        ├── features/
        │   └── build_features.py        # 피처 생성
        ├── modeling/                    # 학습 · 평가 · 예측 · 튜닝
        └── eda/
            └── run_eda.py               # EDA 리포트 생성
```

---

## 🔄 데이터 파이프라인

```
[kbo_scraping.py]         →  data/raw/kbo_{연도}_attendance.csv
[kbo_standings_scrape.py] →  data/external/kbo_standings_daily.csv
[kbo_size.py]             →  data/external/kbo_stadium_info.csv
        ↓
[weather_api.py]          →  data/interim/kbo_*_attendance_weather.csv
        ↓
[preprocess_attendance_weather.py] → data/processed/final_dataset.csv
        ↓
[build_features.py]       →  data/processed/kbo_train_ready.csv
        ↓
[train_model.py]          →  models/attendance_rf_pipeline.joblib
        ↓
[evaluate_model.py]       →  models/eval_report.json
        ↓
[streamlit_app.py]        →  🌐 웹 UI에서 실시간 예측
```

---

## 🚀 실행 방법

### 사전 준비

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

### 권장 실행 순서

```bash
cd machine-learning-project

# 1) 원시 데이터 수집 (Selenium + Chrome 필요)
python3 scripts/data_collection/kbo_scraping.py
python3 scripts/data_collection/kbo_standings_scrape.py

# 2) 구장 수용 인원 CSV 생성
python3 scripts/data_collection/kbo_size.py

# 3) 기상 데이터 병합 → interim
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

> 💡 원시 CSV가 이미 있는 경우 1)~3) 단계를 건너뛰고 `interim` 또는 `processed` 단계부터 시작할 수 있습니다.

### Streamlit 웹앱 실행

```bash
cd machine-learning-project
streamlit run scripts/app/streamlit_app.py
```

> 웹앱 실행 전에 반드시 `train_model.py`로 모델을 먼저 학습해 두세요.  
> 앱은 `kbo_2025_attendance_weather.csv` 파일을 `scripts/app/` 또는 `data/` 경로에서 자동 탐색합니다.

---

## 🖥️ Streamlit UI 기능

| 기능 | 설명 |
|------|------|
| 사이드바 입력 | 경기 날짜, 홈·원정팀, 구장 선택 |
| 기상 슬라이더 | 기온, 강수량 등 기상 조건 조정 |
| KBO 자동 반영 | 최근 경기 관중 데이터 자동 업데이트 옵션 |
| 모델 선택 | RandomForest 파이프라인 사용 여부 토글 |
| 예측 시각화 | 예측 결과 및 관련 통계 차트 제공 |

---

## 📊 주요 스크립트

| 스크립트 | 역할 |
|----------|------|
| `data_collection/kbo_scraping.py` | 일별 관중·경기 메타 크롤링 |
| `data_collection/kbo_standings_scrape.py` | 일자별 팀 순위·승률 스냅샷 수집 |
| `data_collection/kbo_size.py` | 구단·구장별 최대 수용 인원 CSV 생성 |
| `data_collection/weather_api.py` | 기상청 API 연동 → interim CSV 생성 |
| `data_collection/fetch_recent_crowd.py` | 구장별 최근 N경기 관중 추출 |
| `preprocessing/preprocess_attendance_weather.py` | interim 병합 → `final_dataset.csv` |
| `features/build_features.py` | 학습용 피처 생성 → `kbo_train_ready.csv` |
| `modeling/train_model.py` | 시간 순 홀드아웃 검증 + RF 파이프라인 학습 |
| `modeling/evaluate_model.py` | 테스트 구간 재평가 및 리포트 저장 |
| `modeling/predict.py` | 저장된 모델로 배치 예측 |
| `modeling/tune_hyperparams.py` | Optuna + TimeSeriesSplit 하이퍼파라미터 튜닝 |
| `eda/run_eda.py` | 탐색적 데이터 분석 리포트 생성 |
| `app/streamlit_app.py` | 관람 수요 예측 웹앱 |
| `common/stadium_aliases.py` | 구장 이름 별칭 공통 정의 |

---

## ⚙️ 모델 정보

- **알고리즘**: RandomForest Regressor (scikit-learn Pipeline)
- **검증 방식**: 시간 순 홀드아웃 (Time-based holdout)
- **튜닝**: Optuna + TimeSeriesSplit (선택 사항)
- **저장 포맷**: `models/attendance_rf_pipeline.joblib`
- **선택 모델**: LightGBM (`train_model.py` 내 선택적 사용)

---

## 🔑 기상 API 설정

기상청 API 인증키는 현재 **`scripts/data_collection/weather_api.py` 상단 변수**로만 설정되어 있습니다.  
저장소에 키를 커밋하지 않으려면 로컬에서만 수정하거나, 추후 `os.environ` 등으로 분리하는 편이 안전합니다.

---

## 🌿 Git 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 배포용 안정 버전 |
| `develop` | 기능 통합 브랜치 |
| `feat/*` / `feature/*` | 기능별 개발 브랜치 |

---

## 📝 라이선스

저장소 정책에 맞게 `LICENSE` 파일을 추가하세요.

---

## 🙋 문의
팀장:허은준
연락처:enzun123@gmail.com