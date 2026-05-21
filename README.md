# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**🌐 데모:** [kbo-ml-prediction.streamlit.app](https://kbo-ml-prediction.streamlit.app)

---

## 📌 프로젝트 개요

경기 일정·구장·기상·팀 순위를 결합해 **관중 수**를 예측합니다.

- **웹앱:** 예측·혼잡도·운영 액션·동네예보(우천 참고)
- **피처:** 승률·페넌트·매치업·최근 5경기·시즌 진행률·기상 버킷 등
- **모델:** RandomForest(기본) · LightGBM · XGBoost (사이드바에서 선택·평균 앙상블)

repo에 데이터·`.joblib`가 있으면 **수집·학습 없이 웹앱만** 실행할 수 있습니다.

### 📦 저장소에 포함된 것 (앱만 켤 때)

경로는 모두 `machine-learning-project/` 아래입니다.

| 경로 | 설명 |
|------|------|
| `data/processed/kbo_train_ready.csv` | 학습용 피처 |
| `models/attendance_rf_pipeline.joblib` | RandomForest |
| `models/attendance_lgbm_pipeline.joblib` | LightGBM |
| `models/attendance_xgb_pipeline.joblib` | XGBoost |

---

## 📁 폴더 구조 — 꼭 읽기

clone하면 **폴더가 두 단계**입니다.

```
프로젝트 루트/                    ← README.md · requirements.txt (여기서 pip 설치)
├── README.md
├── requirements.txt
├── packages.txt                  ← Streamlit Cloud용 (Windows 로컬 ❌)
└── machine-learning-project/     ← pyproject.toml · data · models · scripts
    ├── pyproject.toml
    ├── data/
    ├── models/
    ├── tests/
    └── scripts/
        └── app/streamlit_app.py   ← 웹앱 진입점
```

| 위치 | 하는 일 |
|------|---------|
| **프로젝트 루트** | `pip install -r requirements.txt` · `python -m streamlit run machine-learning-project\...` |
| **machine-learning-project** | `pip install -e .` · 파이프라인 스크립트 · `pytest` |

> ❌ **프로젝트 루트**에서 `pip install -e .` → `pyproject.toml not found`  
> ❌ **프로젝트 루트**에서 `streamlit run scripts\app\...` → `scripts` 폴더 없음 (하위 폴더 경로 필요)  
> ❌ `python streamlit_app.py` (IDE ▶ 실행) → Streamlit 경고·브라우저 안 열림 → **`streamlit run` 필수**

---

## ⚡ 빠른 시작

**Python 3.10 ~ 3.12** 권장.

### 1) 설치 (프로젝트 루트)

**README.md가 있는 폴더**로 이동한 뒤:

**Windows (PowerShell)**

```powershell
cd C:\경로\Machine-Learning-Project
python -m pip install -U pip
pip install -r requirements.txt
```

**macOS**

```bash
cd ~/경로/Machine-Learning-Project
pip3 install -U pip
pip3 install -r requirements.txt
```

`requirements.txt`가 `machine-learning-project` 패키지를 editable로 설치합니다 (`-e ./machine-learning-project`).

**대안** — `machine-learning-project` 안에서만 설치하고 싶을 때:

```powershell
cd machine-learning-project
pip install -e .
```

### 2) Streamlit 실행 (프로젝트 루트)

`streamlit` 명령이 없을 수 있으므로 **`python -m streamlit`** 을 기본으로 씁니다.

**Windows**

```powershell
cd C:\경로\Machine-Learning-Project
python -m streamlit run machine-learning-project\scripts\app\streamlit_app.py
```

**macOS**

```bash
cd ~/경로/Machine-Learning-Project
python3 -m streamlit run machine-learning-project/scripts/app/streamlit_app.py
```

- 브라우저: `http://localhost:8501`
- `streamlit`이 PATH에 있으면: `streamlit run machine-learning-project\scripts\app\streamlit_app.py` 도 가능
- 사이드바 **ML 알고리즘:** RF · LGBM · XGB 체크 → 예측 **평균**

### 3) (선택) 테스트

```powershell
# 프로젝트 루트
pip install -e "./machine-learning-project[dev]"
cd machine-learning-project
python -m pytest
```

---

## 🪟 Windows · macOS

| 구분 | Windows | macOS |
|------|---------|-------|
| 설치 (권장) | 루트 `pip install -r requirements.txt` | 동일 |
| 웹앱 실행 | `python -m streamlit run machine-learning-project\scripts\app\streamlit_app.py` | `python3 -m streamlit run ...` |
| 파이프라인 | `cd machine-learning-project` 후 `python scripts\...` | `python3 scripts/...` |
| Selenium | Google Chrome + `webdriver-manager` | 동일 |
| `packages.txt` | 사용 안 함 | 사용 안 함 |
| EDA PNG 한글 | `run_eda.py` → `AppleGothic` (맥 전용) · 깨질 수 있음 | ✅ |

---

## 🤖 모델 학습 (RF · LGBM · XGB)

`benchmark_models.py` **한 번**으로 3개 `.joblib` 생성.

```powershell
# 프로젝트 루트
pip install -e "./machine-learning-project[benchmark]"
cd machine-learning-project
python scripts\modeling\benchmark_models.py
```

| 목적 | 스크립트 |
|------|----------|
| RF만 | `scripts/modeling/train_model.py` |
| 3개 + 비교 | `scripts/modeling/benchmark_models.py` |

**벤치마크 MAE:** Dummy 4,671 → RF 1,964 → LGBM **1,911** → XGB 1,928

---

## 🔄 전체 파이프라인

**`machine-learning-project` 폴더**에서 실행 (`pip install`은 이미 완료된 상태).

| 단계 | 스크립트 | 비고 |
|------|----------|------|
| 1 | `kbo_scraping.py`, `kbo_standings_scrape.py` | Chrome (전자만) |
| 2 | `kbo_size.py` | |
| 3 | `weather_api.py` | `KMA_APIHUB_AUTH_KEY` |
| 4~8 | preprocess → features → (eda) → train/benchmark → (evaluate) | |

**Windows 예시**

```powershell
cd machine-learning-project
$env:KMA_APIHUB_AUTH_KEY = "your_key"

python scripts\data_collection\kbo_scraping.py
python scripts\data_collection\kbo_standings_scrape.py
python scripts\data_collection\kbo_size.py
python scripts\data_collection\weather_api.py
python scripts\preprocessing\preprocess_attendance_weather.py
python scripts\features\build_features.py
python scripts\modeling\benchmark_models.py
```

---

## 🔑 환경 변수 · Secrets

| 변수 | 용도 |
|------|------|
| `KMA_APIHUB_AUTH_KEY` | 기상 API · 동네예보 |
| `STREAMLIT_WEB_RECENT=0` | 최근 5경기 크롤 끔 |

Windows: `$env:KMA_APIHUB_AUTH_KEY = "키"` · macOS: `export KMA_APIHUB_AUTH_KEY="키"`

로컬: `machine-learning-project/.streamlit/secrets.toml`

---

## 📦 선택: 가상환경

**프로젝트 루트**에 venv를 두는 예:

```powershell
cd C:\경로\Machine-Learning-Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run machine-learning-project\scripts\app\streamlit_app.py
```

`Activate.ps1` 오류: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` 또는 cmd + `.venv\Scripts\activate.bat`

---

## 🪟 문제 해결

| 증상 | 해결 |
|------|------|
| `pyproject.toml not found` / `does not appear to be a Python project` | 루트에서 **`pip install -r requirements.txt`** (❌ `pip install -e .`) |
| `streamlit` is not recognized | **`python -m streamlit run ...`** |
| `scripts\app` 경로 없음 | 루트에 `scripts` 없음 → **`machine-learning-project\scripts\app\...`** |
| `missing ScriptRunContext` / `bare mode` | IDE에서 `python streamlit_app.py` 실행 중 → **`python -m streamlit run`** 사용 |
| `ModuleNotFoundError: common` | `pip install -r requirements.txt` (루트) 또는 `pip install -e .` (`machine-learning-project`) |
| `python` 없음 | `py -3.12 -m pip install -r requirements.txt` |
| PowerShell `&&` 오류 | 한 줄씩 또는 `;` |
| RF/LGBM/XGB 비활성 | `.joblib` 없음 → `benchmark_models.py` |
| EDA 한글 □□□ (Windows) | `eda_summary.md`는 정상 · PNG만 `run_eda.py` 폰트 이슈 |

---

## ☁️ Streamlit Cloud

| 항목 | 값 |
|------|-----|
| Main file | `machine-learning-project/scripts/app/streamlit_app.py` |
| Python | 3.12 권장 |

루트 `requirements.txt` · `packages.txt`(Linux). Secrets: `KMA_APIHUB_AUTH_KEY` (선택).

---

## 📊 주요 스크립트 · 브랜치 · 팀

| `scripts/` | 역할 |
|------------|------|
| `app/streamlit_app.py` | 메인 웹앱 |
| `modeling/benchmark_models.py` | 3모델 학습 |
| `features/build_features.py` | 피처 |
| `data_collection/kbo_scraping.py` | 관중 크롤 |

| 브랜치 | 용도 |
|--------|------|
| `main` | Cloud 배포 |
| `develop` | 기능 통합 |

| 팀장 | 허은준 (enzun123) — enzun123@gmail.com |
| 팀원 | 김지원, 이승민, 최종원 |

교육·팀 프로젝트 (KBO 관중 예측 ML).
