# KBO 관중 예측 서비스 UI (Streamlit)

**브랜치:** `feat/streamlit-ui`

학습된 머신러닝 파이프라인(`joblib`)을 써서 사용자가 경기 조건을 입력하고 관중 수를 예측해 볼 수 있는 웹 대시보드입니다.

## UI 프로젝트 개요

이 브랜치는 파이프라인의 최종 산출물인 **학습된 모델**을 브라우저에서 쓰기 쉬운 형태로 붙이는 것을 목표로 합니다. 터미널에서 긴 스크립트를 돌리지 않고도 날짜·대진·구장·날씨를 바꿔 가며 관중 규모를 시뮬레이션할 수 있습니다.

## 전체 코드 실행 순서

데이터·모델을 **처음부터** 맞춰 나갈 때는 아래 **브랜치(단계) 순서**를 따르는 것을 권장합니다. 앞 단계 산출물이 다음 단계의 입력이 됩니다.

1. `feat/scraping-kbo` — KBO 관중 등 기본 수집  
2. `feat/weather-api` — 기상 데이터 결합  
3. `feat/preprocessing` — 전처리  
4. `feat/stadium-capacity` — 구장 정본 `kbo_stadium_info.csv`  
5. `feat/eda` — 탐색적 분석 (`reports/eda/` 등)  
6. `feat/feature-engineering` — `build_features.py` → `kbo_train_ready.csv`  
7. `feat/ml-modeling` — `train_model.py` / `evaluate_model.py` → `attendance_rf_pipeline.joblib`  
8. `feat/streamlit-ui`

**참고:** 구장별 **최근 N경기 관중** 스크랩 CLI는 `feature/kbo-recent-games-scrape` 계열(`scripts/data_collection/fetch_recent_crowd.py`)에서 다루며, UI에서 옵션으로 호출할 수 있습니다.

## 주요 구조

```text
machine-learning-project/
├── scripts/
│   ├── app/
│   │   └── streamlit_app.py          # 메인 Streamlit 앱
│   ├── data_collection/
│   │   └── fetch_recent_crowd.py   # (옵션) 구장별 최근 N경기 스크랩 — 앱에서 호출
│   └── modeling/
│       └── train_model.py          # 모델 학습 (RF 파이프라인 → joblib)
├── data/
│   ├── external/kbo_stadium_info.csv
│   └── processed/kbo_train_ready.csv
└── models/
    └── attendance_rf_pipeline.joblib
```

## 주요 기능

| 기능 | 설명 |
|------|------|
| 실시간 예측 | 날짜·홈/원정·구장을 고르고, 옵션으로 RF 파이프라인으로 예측 |
| 날씨 시뮬레이션 | 기온·강수·습도 슬라이더로 조건 변경 |
| 정원 대비 표시 | 구장 수용 인원(`kbo_stadium_info.csv` 등)과 예측값 비교 |
| 최근 경기 | 옵션으로 KBO 기록실 스크랩 연동(구장·일자 기준, 캐시 TTL 설정 가능) |
| 차트·통계 | 과거 CSV 기반 참고 그래프·표 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install streamlit pandas numpy matplotlib joblib scikit-learn
```

최근 경기 자동 스크랩을 쓰려면 추가로:

```bash
pip install beautifulsoup4 selenium webdriver-manager
```

### 2. 모델·데이터 확인

- **모델:** `machine-learning-project/models/attendance_rf_pipeline.joblib`  
  없으면 저장소 루트에서:

  ```bash
  python3 machine-learning-project/scripts/modeling/train_model.py
  ```

- **RF 예측 시:** `machine-learning-project/data/processed/kbo_train_ready.csv` 필요.
- **앱 기본 CSV:** `kbo_2025_attendance_weather.csv` (앱이 여러 경로에서 탐색). 필수 컬럼 예: `경기날짜`, `홈팀`, `방문팀`, `구장`, `관중수` 등.
- **구장 정원:** `machine-learning-project/data/external/kbo_stadium_info.csv` 권장 (`구장`, `최대수용인원`).

### 3. 웹앱 실행

저장소 루트(`Machine-Learning-Project`)에서:

```bash
streamlit run machine-learning-project/scripts/app/streamlit_app.py
```

터미널에 나온 Local URL(보통 `http://localhost:8501`)로 접속합니다.

### (옵션) 환경 변수

- `KBO_APP_RECENT_TTL_SEC` — 최근 경기 스크랩 결과 캐시 TTL(초), 기본 `900`
- `KBO_SCRAPE_HEADLESS` — `0`이면 헤드리스 끔(디버깅용)

## 데이터 흐름 (UI)

1. 사용자가 사이드바에서 경기 날짜·매치업·구장·날씨 등 입력  
2. RF 사용 시 `kbo_train_ready.csv`에서 템플릿 행을 고르고 입력값으로 피처 행 구성  
3. `attendance_rf_pipeline.joblib` 로드 후 추론  
4. 메인 영역에 예측값·정원 대비·차트 등 표시  

## 주의 사항

- **joblib / sklearn 버전:** 학습할 때와 실행할 때 `scikit-learn` 버전 차이가 크면 로드 오류가 날 수 있습니다.
- **구장 이름:** `common/stadium_aliases.py`와 학습 데이터의 구장 표기가 맞아야 합니다.
- **최근 경기 스크랩:** Chrome/Chromium·네트워크 환경에 따라 실패할 수 있으며, 실패 시 앱은 로그만 남기고 빈 결과로 동작할 수 있습니다.

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  

