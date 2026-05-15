# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**현재 브랜치:** `feat/streamlit-ux-and-safety` — Streamlit UX·안전·스타일 분리

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest**로 KBO 관중 수를 예측합니다. 이 브랜치는 **Streamlit 웹앱**의 사용성·표시 안전·코드 구조를 정리합니다.

**이 브랜치에서 추가·정리한 내용**
- `session_state` — 관중 CSV·구장 정원 캐시, 구장 변경 시 홈팀 자동 연동
- `scripts/app/styles/app.css` — UI 스타일을 Python에서 분리
- `common/congestion_levels.py` — 혼잡도(%) → 운영 단계·액션 메시지
- `html.escape` — 사용자 입력·API 메시지의 HTML 삽입 방지
- `_scalar_rain_bucket_ml()` — `build_features`와 동일한 강수 구간(`No_Rain` ~ `Rain_5mm_plus`)
- API 키 마스킹 — 오류 URL·로그의 `authKey`/`serviceKey` 숨김

(동네예보 typ02·우천 참고 UI, RF 피처 중요도 날씨 2그룹 합산 등은 선행 커밋에 포함)

---

## 📁 저장소 구조 (앱 관련)

```
machine-learning-project/
└── scripts/
    ├── app/
    │   ├── streamlit_app.py       # ★ 웹 UI
    │   └── styles/
    │       └── app.css            # ★ 커스텀 스타일
    ├── common/
    │   ├── congestion_levels.py   # ★ 혼잡도 분류
    │   ├── kma_vilage_fcst.py     # 동네예보 typ02
    │   ├── stadium_aliases.py
    │   └── kbo_regular_start_time.py
    ├── data_collection/
    │   └── fetch_recent_crowd.py  # 최근 5경기 (앱 연동)
    ├── modeling/
    ├── features/
    └── data/
        ├── interim/               # 관중·기상 CSV
        ├── processed/             # kbo_train_ready.csv
        └── models/                # attendance_rf_pipeline.joblib
```

---

## 🚀 실행 방법

### 사전 준비

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

**학습 산출물 (RF 사용 시)**
- `models/attendance_rf_pipeline.joblib`
- `data/processed/kbo_train_ready.csv`
- `data/interim/kbo_2025_attendance_weather.csv` 등 (앱이 여러 경로에서 탐색)

### Streamlit 실행 (이 브랜치 핵심)

```bash
cd machine-learning-project
export KMA_APIHUB_AUTH_KEY="발급받은_키"
streamlit run scripts/app/streamlit_app.py
```

| 환경변수 | 설명 |
|----------|------|
| `KMA_APIHUB_AUTH_KEY` | 동네예보 typ02 (또는 `.streamlit/secrets.toml`) |
| `STREAMLIT_WEB_RECENT=0` | KBO 최근 5경기 GraphDaily 수집 끄기 (기본: 켜짐) |
| `STREAMLIT_DEBUG_WEATHER=1` | 동네예보 API 디버그 패널 |
| `KBO_APP_RECENT_TTL_SEC` | 최근 경기 캐시 TTL(초, 기본 `900`) |
| `KBO_SCRAPE_HEADLESS=0` | 최근 경기 크롤 시 브라우저 표시 |

전체 파이프라인(수집 → 전처리 → 피처 → 학습)은 `develop` 또는 다른 `feat/*` 브랜치 README를 참고하세요.

---

## 🖥️ Streamlit UI 기능

| 기능 | 설명 |
|------|------|
| 사이드바 | 경기 날짜, 구장·홈·원정, 기온·**일 강수(mm)**·습도 |
| 구장 연동 | `session_state`로 구장 변경 시 기본 홈팀 자동 설정 |
| 예측 모드 | **RandomForest** 또는 과거 CSV 평균 + 날씨 룰(휴리스틱) |
| 강수 버킷 | 슬라이더(mm) → `rain_bucket` 등 RF 입력과 학습 파이프라인 정합 |
| 피처 중요도 | RF 사용 시 막대 그래프(날씨 세부 2그룹 합산) |
| 혼잡도 | 수용률 % → LOW / NORMAL / HIGH + 운영 액션 카드 (`congestion_levels`) |
| 최근 5경기 | 옵션 시 GraphDaily (Selenium, TTL 캐시) |
| 동네예보 | 개시 3시간 전 RN1/POP, 우천 취소 **참고** (관중 예측과 분리) |
| 안전 | `html.escape`, API 키 URL 마스킹 |

---

## 🧩 주요 모듈

| 파일 | 역할 |
|------|------|
| `app/streamlit_app.py` | UI·예측·차트·기상 참고 |
| `app/styles/app.css` | 레이아웃·카드·우천 경고 박스 스타일 |
| `common/congestion_levels.py` | `classify_congestion_pct()` — 50% / 80% 구간 |
| `common/kma_vilage_fcst.py` | `forecast_ref_for_rain_cancel_rules`, `rainout_cancel_guidance` |
| `data_collection/fetch_recent_crowd.py` | 구장별 최근 N경기 |

---

## ⚙️ 모델·예측 (요약)

- **RF**: `kbo_train_ready.csv` 유사 행 + 사이드바 기상 덮어쓰기 → `attendance_rf_pipeline.joblib`
- **휴리스틱**: 매치업/홈/원정/구장 평균, 강수·고습도 보정
- **혼잡도**: `예상 관중 / 구장 정원 × 100` → `congestion_levels` 액션 플랜

| 지표 (테스트 구간 참고) | RandomForest |
|-------------------------|--------------|
| MAE | **~1,943** |
| R² | **~0.73** |

---

## 🔑 API·키

| 용도 | 설정 |
|------|------|
| Streamlit 동네예보(typ02) | `KMA_APIHUB_AUTH_KEY` |
| 배치 기상(typ01) | `weather_api.py`의 `AUTH_KEY` (로컬만, 커밋 금지) |

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `feat/streamlit-ux-and-safety` | **현재** — Streamlit UX·CSS·혼잡도·escape |
| `feat/build-features-refactor` | 피처 파이프라인 |
| `feat/weather-kma-refactor` | KMA env 통일 |
| `feat/kbo-scraping-and-standings` | 데이터 수집 |
| `develop` | 기능 통합 |

---

## 🙋 문의

- 팀장: 허은준
- 연락처: enzun123@gmail.com
