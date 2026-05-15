# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**현재 브랜치:** `feat/security-api-and-exceptions` — API 키 환경변수화·예외 처리·RF 폴백 안내

---

## 📌 프로젝트 개요

경기 일정, 구장 정보, 기상 데이터, 팀 순위를 결합해 **RandomForest**로 KBO 관중 수를 예측합니다. 이 브랜치는 **보안(키 노출 방지)** 과 **실패 시 동작·사용자 안내**를 정리합니다.

**이 브랜치에서 추가·정리한 내용**
- **API 키**: typ01(`weather_api.py`)·typ02(`kma_vilage_fcst.py`) 모두 `KMA_APIHUB_AUTH_KEY` 환경변수만 사용 (코드에 키 하드코딩 제거)
- **기상 배치**: 키 없으면 조기 종료·요청 건너뜀; `requests.RequestException` 시 빈 결과로 계속
- **캐시**: `weather_cache.json` 읽기 실패 시 `JSONDecodeError`/`OSError` 처리 후 재생성
- **크롤링**: `NoSuchElementException` / `WebDriverException` 분리 처리
- **동네예보**: `redact_api_secrets()` — 오류·URL의 `authKey`/`serviceKey` 마스킹
- **Streamlit**: RF 예측 실패·산출물 없음 시 **휴리스틱 폴백** 메시지, API 오류 문자열 마스킹

---

## 📁 저장소 구조 (관련 파일)

```
machine-learning-project/
└── scripts/
    ├── data_collection/
    │   ├── weather_api.py          # typ01 — KMA_APIHUB_AUTH_KEY
    │   └── kbo_scraping.py         # Selenium 예외 분리
    ├── common/
    │   └── kma_vilage_fcst.py      # typ02 — redact_api_secrets
    └── app/
        └── streamlit_app.py        # RF 폴백·_redact_kma_secret_str
```

---

## 🔄 데이터 파이프라인

```
[kbo_scraping.py]         →  data/raw/
[kbo_standings_scrape.py] →  data/external/kbo_standings_daily.csv
[kbo_size.py]             →  data/external/kbo_stadium_info.csv
        ↓
[weather_api.py]          →  data/interim/   ← KMA_APIHUB_AUTH_KEY 필수
        ↓
[preprocess → build_features → train_model]
        ↓
[streamlit_app.py]
```

---

## 🚀 실행 방법

### 사전 준비

```bash
pip install pandas numpy requests beautifulsoup4 selenium webdriver-manager \
            scikit-learn joblib matplotlib seaborn streamlit optuna lightgbm
```

### API 키 설정 (필수)

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
```

| 용도 | 모듈 |
|------|------|
| 일별 관측 typ01 | `weather_api.py` |
| 동네예보 typ02 | `kma_vilage_fcst.py`, Streamlit |

- 키가 **비어 있으면** `weather_api.py`는 에러 로그 후 **종료** (interim 미생성)
- Streamlit은 secrets: `.streamlit/secrets.toml`의 `KMA_APIHUB_AUTH_KEY` 가능
- **`.env`·키 문자열을 Git에 커밋하지 마세요**

### 기상 병합

```bash
cd machine-learning-project
export KMA_APIHUB_AUTH_KEY="발급받은_키"
python3 scripts/data_collection/weather_api.py
```

### Streamlit

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
streamlit run scripts/app/streamlit_app.py
```

**RF 사용 시 필요 파일**
- `models/attendance_rf_pipeline.joblib`
- `data/processed/kbo_train_ready.csv`

없거나 예측 오류 시 앱은 **휴리스틱(과거 평균)** 으로 표시하고 화면에 안내합니다.

| 환경변수 | 설명 |
|----------|------|
| `KMA_APIHUB_AUTH_KEY` | 기상청 API허브 (typ01·typ02) |
| `STREAMLIT_WEB_RECENT=0` | 최근 5경기 자동 수집 끄기 |
| `STREAMLIT_DEBUG_WEATHER=1` | 동네예보 디버그 패널 |
| `KBO_SCRAPE_HEADLESS=0` | 크롤 시 브라우저 표시 |

---

## 🛡️ 예외·폴백 동작

| 구간 | 동작 |
|------|------|
| `weather_api` 키 없음 | `main()` 조기 return, 로그에 env 이름 안내 |
| `fetch_weather` HTTP 실패 | `RequestException` 로그 후 `{}` 반환 |
| `weather_cache` 손상 | 경고 후 빈 캐시로 재수집 |
| `kbo_scraping` 페이지 이동 | `NoSuchElementException` → 마지막 페이지; `WebDriverException` → 로그 후 중단 |
| `kma_vilage_fcst` 오류 메시지 | `redact_api_secrets` 적용 |
| Streamlit RF `predict` 실패 | 로그 + 캡션 «휴리스틱 유지» |
| RF 체크 ON, 모델 파일 없음 | 캡션 «휴리스틱만» + 체크박스 도움말 |

---

## 🖥️ Streamlit (요약)

- RandomForest / 휴리스틱, 동네예보·우천 참고, 피처 중요도(날씨 2그룹)
- 사용자·API 텍스트는 `html.escape` / `_redact_kma_secret_str` 처리

---

## ⚙️ 모델 정보

- RandomForest, 시간 순 홀드아웃
- 테스트 구간 참고: MAE **~1,943**, R² **~0.73**

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `feat/security-api-and-exceptions` | **현재** — API env·예외·RF 안내 |
| `feat/weather-kma-refactor` | 동네예보 ref 단계 분리 |
| `feat/streamlit-ux-and-safety` | CSS·session_state·혼잡도 모듈 |
| `develop` | 기능 통합 |

---

## 🙋 문의

- 팀장: 허은준
- 연락처: enzun123@gmail.com
