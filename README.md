# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**🌐 데모:** [kbo-ml-prediction.streamlit.app](https://kbo-ml-prediction.streamlit.app)

---

## 📌 프로젝트 개요

경기 일정·구장·기상·팀 순위를 결합해 **관중 수**를 예측합니다.

- **웹앱:** 예측·혼잡도·운영 액션·동네예보(우천 참고)
- **피처:** 승률·페넌트·매치업·최근 5경기·시즌 진행률·기상 버킷 등
- **모델:** RandomForest(기본) · LightGBM · XGBoost (사이드바에서 선택·평균 앙상블)

repo에 데이터·`.joblib`가 있으면 **수집·학습 없이 Streamlit만** 실행할 수 있습니다.

### 📦 저장소에 포함된 것 (앱만 켤 때)

`machine-learning-project/` 기준:

| 경로 | 설명 |
|------|------|
| `data/processed/kbo_train_ready.csv` | 학습용 피처 테이블 |
| `models/attendance_rf_pipeline.joblib` | RandomForest |
| `models/attendance_lgbm_pipeline.joblib` | LightGBM |
| `models/attendance_xgb_pipeline.joblib` | XGBoost |
| `data/raw/`, `data/interim/`, `data/external/` | 관중·기상·구장·순위 등 |

---

## 📁 폴더 구조 (필수)

```
<clone-root>/                      ← git clone 루트 (README 위치)
│                                    예: Machine-Learning-Project-1
├── README.md
├── requirements.txt               ← Streamlit Cloud (루트 pip)
├── packages.txt                   ← Linux apt 전용 (Windows 로컬 ❌)
└── machine-learning-project/      ← ★ pip · streamlit · pytest · 스크립트
    ├── pyproject.toml
    ├── data/
    ├── models/
    ├── reports/
    ├── tests/
    └── scripts/
        ├── app/streamlit_app.py
        ├── common/
        ├── data_collection/
        ├── preprocessing/
        ├── features/
        ├── modeling/
        └── eda/
```

> **주의:** `pyproject.toml`은 **루트가 아니라** `machine-learning-project/` 안에 있습니다.  
> clone 루트에서 `pip install -e .` 하면 `pyproject.toml not found` 오류가 납니다.

---

## 🪟 Windows · macOS 호환

코드는 **`pathlib` 경로**·**UTF-8( BOM ) CSV**·**Selenium Chrome 폴백**으로 Windows/macOS 공통 설계입니다.

| 구분 | Windows | macOS |
|------|---------|-------|
| 경로 | `\` · `pathlib` ✅ | `/` ✅ |
| `pip install -e .` | `machine-learning-project`에서 실행 | 동일 |
| Streamlit | ✅ (한글: Malgun·Nanum 폴백) | ✅ |
| pytest | ✅ (`pip install -e ".[dev]"`) | ✅ |
| Selenium 크롤 | **Google Chrome** + `webdriver-manager` | 동일 |
| `packages.txt` | **사용 안 함** (Cloud/Linux) | 사용 안 함 |
| EDA 차트 한글 | ⚠️ `run_eda.py`만 `AppleGothic` 고정 → PNG 한글 깨질 수 있음 | ✅ |
| EDA 요약 `eda_summary.md` | ✅ | ✅ |

---

## ⚡ 빠른 시작

**Python 3.10 ~ 3.12** 권장. 3.14 등 최신 버전은 일부 패키지(wheel) 호환 문제가 있을 수 있습니다.

### 1) 설치 (한 번)

**Windows (PowerShell)** — `<clone-root>`를 실제 클론 폴더로 바꿉니다.

```powershell
cd <clone-root>\machine-learning-project
python -m pip install -U pip
pip install -e .
```

**macOS**

```bash
cd <clone-root>/machine-learning-project
pip3 install -U pip
pip3 install -e .
```

| 항목 | 설명 |
|------|------|
| `pip install -e .` | `machine-learning-project`를 현재 Python에 설치 (`common`, `modeling` 등 import 가능) |
| 가상환경 | **필수 아님** — 패키지 충돌 시 [선택: venv](#-선택-가상환경-venv) |
| PowerShell 5.x | `&&` 대신 **한 줄씩** 또는 `;` (7+만 `&&`) |

### 2) Streamlit 실행

**Windows**

```powershell
cd <clone-root>\machine-learning-project
streamlit run scripts\app\streamlit_app.py
```

**macOS**

```bash
cd <clone-root>/machine-learning-project
streamlit run scripts/app/streamlit_app.py
```

- 브라우저: `http://localhost:8501`
- `streamlit` 없음: `python -m streamlit run scripts/app/streamlit_app.py`
- 사이드바 **ML 알고리즘:** RF · LGBM · XGB 체크 → 예측 **평균**

### 3) (선택) 테스트

```powershell
# Windows — machine-learning-project 폴더
pip install -e ".[dev]"
python -m pytest
```

```bash
# macOS
pip install -e ".[dev]"
pytest
```

---

## 🤖 모델 학습 (3개 한 번에)

**3번 따로 학습할 필요 없음.** `benchmark_models.py` 한 번으로 RF · LGBM · XGB `.joblib` 생성.

```powershell
# Windows
cd <clone-root>\machine-learning-project
pip install -e ".[benchmark]"
python scripts\modeling\benchmark_models.py
```

```bash
# macOS
pip install -e ".[benchmark]"
python3 scripts/modeling/benchmark_models.py
```

| 목적 | 스크립트 |
|------|----------|
| RF만 | `scripts/modeling/train_model.py` |
| 3개 + 벤치마크 | `scripts/modeling/benchmark_models.py` |
| RF 평가 | `scripts/modeling/evaluate_model.py` (선택) |
| 하이퍼튜닝 | `scripts/modeling/tune_hyperparams.py` (선택) |

**벤치마크 (테스트 287경기, MAE):** Dummy 4,671 → RF **1,964** → LGBM **1,911** → XGB 1,928 · R² LGBM **0.77**

---

## 🔄 전체 파이프라인 (처음부터)

`machine-learning-project`에서 실행. `pip install -e .` 후 **`PYTHONPATH`는 보통 불필요**합니다.

| 단계 | 스크립트 | 비고 |
|------|----------|------|
| 1 | `data_collection/kbo_scraping.py` | Chrome + Selenium |
| 1 | `data_collection/kbo_standings_scrape.py` | HTTP만 (Selenium ❌) |
| 2 | `data_collection/kbo_size.py` | 구장 정원 CSV |
| 3 | `data_collection/weather_api.py` | `KMA_APIHUB_AUTH_KEY` |
| 4 | `preprocessing/preprocess_attendance_weather.py` | |
| 5 | `features/build_features.py` | |
| 6 | `eda/run_eda.py` | 선택 · Windows PNG 한글 주의 |
| 7 | `modeling/train_model.py` 또는 `benchmark_models.py` | |
| 8 | `modeling/evaluate_model.py` | 선택 (RF) |

**Windows (PowerShell) 예시**

```powershell
cd <clone-root>\machine-learning-project
$env:KMA_APIHUB_AUTH_KEY = "your_key"   # 3단계만

python scripts\data_collection\kbo_scraping.py
python scripts\data_collection\kbo_standings_scrape.py
python scripts\data_collection\kbo_size.py
python scripts\data_collection\weather_api.py
python scripts\preprocessing\preprocess_attendance_weather.py
python scripts\features\build_features.py
python scripts\modeling\benchmark_models.py
```

**macOS 예시**

```bash
cd <clone-root>/machine-learning-project
export KMA_APIHUB_AUTH_KEY="your_key"

python3 scripts/data_collection/kbo_scraping.py
# ... 동일 순서
python3 scripts/modeling/benchmark_models.py
```

**Selenium (Windows):** `kbo_scraping.py`는 Linux `/usr/bin/chromium`을 찾지 못하면 **로컬 Chrome + `webdriver-manager`** 를 씁니다. [Google Chrome](https://www.google.com/chrome/) 설치 필요.

---

## 🔑 환경 변수 · Secrets

| 변수 | 용도 |
|------|------|
| `KMA_APIHUB_AUTH_KEY` | `weather_api.py`(관측) · Streamlit 동네예보(typ02) |
| `STREAMLIT_WEB_RECENT=0` | KBO 최근 5경기 자동 크롤 끔 |
| `STREAMLIT_DEBUG_WEATHER=1` | 동네예보 API 디버그 패널 |

| OS | 설정 예 |
|----|---------|
| Windows (PowerShell) | `$env:KMA_APIHUB_AUTH_KEY = "키"` |
| Windows (cmd) | `set KMA_APIHUB_AUTH_KEY=키` |
| macOS | `export KMA_APIHUB_AUTH_KEY="키"` |

- **로컬:** `machine-learning-project/.streamlit/secrets.toml`
- **Cloud:** 앱 Settings → Secrets

```toml
KMA_APIHUB_AUTH_KEY = "발급받은_키"
```

---

## 📦 선택: 가상환경 (venv)

다른 과제와 **pandas / scikit-learn** 버전이 섞일 때만 사용합니다.

**Windows**

```powershell
cd <clone-root>\machine-learning-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
pip install -e ".[dev]"
streamlit run scripts\app\streamlit_app.py
```

**macOS**

```bash
cd <clone-root>/machine-learning-project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

`Activate.ps1` 거부 시: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` 또는 **cmd** + `.venv\Scripts\activate.bat`

---

## 🪟 문제 해결

| 증상 | 해결 |
|------|------|
| `pyproject.toml not found` | **`machine-learning-project`로 `cd`** 후 `pip install -e .` |
| `ModuleNotFoundError: common` / `modeling` | 위 폴더에서 `pip install -e .` |
| `cd Machine-Learning-Project\...` 실패 | 이미 clone 루트면 `cd machine-learning-project` 만 |
| `python` 없음 (Windows) | `py -3.12` 사용 |
| `streamlit` 없음 | `python -m streamlit run scripts/app/streamlit_app.py` |
| PowerShell `&&` 오류 | 줄 단위 실행 또는 `;` |
| RF/LGBM/XGB 체크박스 비활성 | 해당 `.joblib` 없음 → `benchmark_models.py` 또는 `train_model.py` |
| 크롤링 실패 | Chrome 설치 · 방화벽 · `STREAMLIT_WEB_RECENT=0`으로 앱만 사용 |

**루트에서 Cloud와 동일 설치 (선택)**

```powershell
cd <clone-root>
pip install -r requirements.txt
```

---

## ☁️ Streamlit Cloud

| 항목 | 값 |
|------|-----|
| Repository | `enzun123/Machine-Learning-Project` |
| Main file | `machine-learning-project/scripts/app/streamlit_app.py` |
| Branch | `main` (또는 `develop`) |
| Python | **3.12** 권장 |

- 루트 `requirements.txt` — `pip install -e ./machine-learning-project`
- 루트 `packages.txt` — `fonts-nanum`, `chromium`, `chromium-driver` (**Linux 전용**, Windows 로컬과 무관)
- Secrets: `KMA_APIHUB_AUTH_KEY` (선택)

| 기능 | 로컬 | Cloud |
|------|------|-------|
| RF/LGBM/XGB 예측 | ✅ | ✅ |
| 동네예보 | API 키 필요 | Secrets |
| KBO 최근 5경기 크롤 | Chrome (기본 ON) | Chromium (`packages.txt`) · 불안정 시 `STREAMLIT_WEB_RECENT=0` |

---

## 🖥️ Streamlit UI 요약

| 기능 | 설명 |
|------|------|
| 사이드바 입력 | 날짜·구장·팀·기온·강수·습도 |
| ML 알고리즘 | RF / LGBM / XGB 다중 선택 → **평균** |
| 혼잡도 | 수용률 → LOW / NORMAL / HIGH + 운영 안내 |
| CSV 일괄 예측 | `csv_batch_predict_ui.py` — 일정 CSV 업로드 |
| 동네예보 | 개시 3시간 전 RN1/POP (예측값과 분리) |
| 최근 5경기 | Selenium (로컬 Chrome / Cloud chromium) |

> **2026·미래 일정:** 모델은 2024–25 시즌 학습. 분포가 다르면 오차가 커질 수 있습니다.

---

## 📊 주요 스크립트

| 경로 (`scripts/`) | 역할 |
|-------------------|------|
| `app/streamlit_app.py` | 메인 웹앱 |
| `app/csv_batch_predict_ui.py` | CSV 일괄 예측 UI |
| `modeling/benchmark_models.py` | 3모델 학습·비교 |
| `modeling/train_model.py` | RF 파이프라인 |
| `modeling/batch_predict.py` | 배치 추론 |
| `features/build_features.py` | 학습 피처 |
| `data_collection/kbo_scraping.py` | 관중 크롤 (Selenium) |
| `data_collection/weather_api.py` | 기상 관측 API |
| `common/kma_vilage_fcst.py` | 동네예보 typ02 |
| `common/congestion_levels.py` | 혼잡도 구간 |
| `eda/run_eda.py` | EDA 리포트·차트 |

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `main` | Streamlit Cloud 배포 |
| `develop` | 기능 통합 |
| `feat/*` | 기능별 개발 |

---

## 👥 팀 · 문의

| 역할 | 이름 |
|------|------|
| 팀장 | 허은준 (enzun123) — enzun123@gmail.com |
| 팀원 | 김지원, 이승민, 최종원 |

교육·팀 프로젝트 (KBO 관중 예측 ML).
