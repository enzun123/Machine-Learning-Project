# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수 예측 — 데이터 파이프라인 + Streamlit 웹앱

**🌐 데모:** [kbo-ml-prediction.streamlit.app](https://kbo-ml-prediction.streamlit.app)

---

## 목차

- [공통 (먼저 읽기)](#공통-먼저-읽기)
- [macOS 실행 가이드](#macos-실행-가이드)
- [Windows 실행 가이드](#windows-실행-가이드)
- [3개 모델 (RF · LGBM · XGB)](#3개-모델-rf--lgbm--xgb)
- [Streamlit Cloud](#streamlit-cloud)
- [모델 성능 · 스크립트 · 팀](#모델-성능--스크립트--팀)

---

## 공통 (먼저 읽기)

### 프로젝트 요약

경기 일정·구장·기상·순위로 **관중 수**를 예측합니다. Streamlit에서 예측·혼잡도·동네예보(우천 참고)를 볼 수 있습니다.

- **모델:** RandomForest(기본) · LightGBM · XGBoost (UI에서 선택, 다중 선택 시 **평균**)
- **repo에 데이터·joblib가 있으면** 크롤링·학습 없이 **웹앱만** 실행 가능

### 폴더 구조

```
Machine-Learning-Project/          ← git clone 루트
├── requirements.txt               ← Streamlit Cloud
├── packages.txt
└── machine-learning-project/      ← ★ 모든 명령은 여기서 실행
    ├── pyproject.toml             ← pip install -e .
    ├── data/
    ├── models/                    ← *.joblib
    ├── tests/                     ← pytest
    └── scripts/
        └── app/streamlit_app.py
```

| 주의 | 설명 |
|------|------|
| `pip install -e .` 위치 | 반드시 **`machine-learning-project/`** 안에서 (루트 X) |
| `cd` 실수 | clone 직후: `.../Machine-Learning-Project/machine-learning-project` · 이미 루트 안: `cd machine-learning-project` |

### 포함된 파일 (앱만 켤 때)

`machine-learning-project/` 기준:

| 경로 | 설명 |
|------|------|
| `data/processed/kbo_train_ready.csv` | 학습 피처 |
| `models/attendance_rf_pipeline.joblib` | RF |
| `models/attendance_lgbm_pipeline.joblib` | LGBM |
| `models/attendance_xgb_pipeline.joblib` | XGB |

### 데이터 파이프라인

```
kbo_scraping / kbo_standings_scrape → raw, standings
kbo_size → kbo_stadium_info.csv
weather_api → interim (*_weather.csv)
preprocess_attendance_weather → final_dataset.csv
build_features → kbo_train_ready.csv
train_model.py          → RF .joblib 만
benchmark_models.py     → RF + LGBM + XGB .joblib (3개 한 번에)
evaluate_model.py       → eval_report.json (RF)
streamlit_app.py        → 웹 UI
```

### 공통 요구사항

| 항목 | 내용 |
|------|------|
| Python | **3.10 ~ 3.12** (3.14 등 최신 버전은 호환 오류 가능) |
| Chrome | 크롤링·앱 «최근 5경기» 사용 시 |
| 기상 API (선택) | `KMA_APIHUB_AUTH_KEY` |

`pip install -e .` 후에는 대부분 **`PYTHONPATH` 불필요** (`common`, `modeling` 패키지 설치됨).

---

## macOS 실행 가이드

### 1. 설치 (가상환경 없음 · 권장)

터미널 (bash / zsh):

```bash
git clone https://github.com/enzun123/Machine-Learning-Project.git
cd Machine-Learning-Project/machine-learning-project

pip3 install -U pip
pip3 install -e .
```

이미 저장소 루트(`Machine-Learning-Project`)에 있다면:

```bash
cd machine-learning-project
pip3 install -e .
```

### 2. Streamlit 웹앱

```bash
cd machine-learning-project
streamlit run scripts/app/streamlit_app.py
```

- 주소: http://localhost:8501
- `streamlit` 없음: `python3 -m streamlit run scripts/app/streamlit_app.py`

**동네예보 (선택)**

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
```

또는 `machine-learning-project/.streamlit/secrets.toml`:

```toml
KMA_APIHUB_AUTH_KEY = "발급받은_키"
```

**최근 5경기 크롤 끄기 (선택)**

```bash
export STREAMLIT_WEB_RECENT=0
```

### 3. pytest (선택)

```bash
cd machine-learning-project
pip3 install -e ".[dev]"
pytest
```

### 4. 3개 모델 학습 (macOS)

```bash
cd machine-learning-project
pip3 install -e ".[benchmark]"
python3 scripts/modeling/benchmark_models.py
```

Streamlit → 사이드바 **ML 알고리즘** → RF · LGBM · XGB **전부 체크**.

### 5. 전체 파이프라인 (macOS)

`machine-learning-project`에서:

```bash
cd machine-learning-project
pip3 install -e .

export KMA_APIHUB_AUTH_KEY="your_key"   # weather_api.py 만 필수

python3 scripts/data_collection/kbo_scraping.py
python3 scripts/data_collection/kbo_standings_scrape.py
python3 scripts/data_collection/kbo_size.py
python3 scripts/data_collection/weather_api.py
python3 scripts/preprocessing/preprocess_attendance_weather.py
python3 scripts/features/build_features.py
python3 scripts/eda/run_eda.py                              # 선택
python3 scripts/modeling/benchmark_models.py                # 3모델
python3 scripts/modeling/evaluate_model.py                  # 선택 (RF)
python3 scripts/modeling/tune_hyperparams.py --n-trials 50  # 선택
```

### 6. 가상환경 (macOS · 선택)

```bash
cd machine-learning-project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

### 7. macOS 문제 해결

| 증상 | 해결 |
|------|------|
| `pyproject.toml not found` | `machine-learning-project`로 `cd` 후 `pip3 install -e .` |
| `python3` 없음 | `brew install python@3.12` 또는 python.org 설치 |
| `ModuleNotFoundError: common` | `pip3 install -e .` 재실행 |
| Selenium 실패 | Google Chrome 설치 |

---

## Windows 실행 가이드

> 명령은 **`python`**, **`pip`** 기준 (없으면 `py -3.12`, `py -3.12 -m pip`).

### 1. 설치 (가상환경 없음 · 권장)

**PowerShell:**

```powershell
git clone https://github.com/enzun123/Machine-Learning-Project.git
cd Machine-Learning-Project\machine-learning-project

python -m pip install -U pip
pip install -e .
```

이미 저장소 루트에 있다면:

```powershell
cd machine-learning-project
pip install -e .
```

**명령 프롬프트 (cmd)** — `cd`·`pip` 동일, 활성화만 다름 (아래 venv 참고).

### 2. Streamlit 웹앱

**PowerShell:**

```powershell
cd machine-learning-project
streamlit run scripts\app\streamlit_app.py
```

**cmd:**

```cmd
cd machine-learning-project
python -m streamlit run scripts\app\streamlit_app.py
```

- 주소: http://localhost:8501

**동네예보 (선택)**

PowerShell:

```powershell
$env:KMA_APIHUB_AUTH_KEY = "발급받은_키"
```

cmd:

```cmd
set KMA_APIHUB_AUTH_KEY=발급받은_키
```

또는 `machine-learning-project\.streamlit\secrets.toml` (내용은 macOS와 동일).

**최근 5경기 크롤 끄기 (선택)**

```powershell
$env:STREAMLIT_WEB_RECENT = "0"
```

### 3. pytest (선택)

**PowerShell / cmd:**

```powershell
cd machine-learning-project
pip install -e ".[dev]"
pytest
```

또는 `python -m pytest`

### 4. 3개 모델 학습 (Windows)

**PowerShell:**

```powershell
cd machine-learning-project
pip install -e ".[benchmark]"
python scripts\modeling\benchmark_models.py
```

Streamlit → 사이드바 **ML 알고리즘** → RF · LGBM · XGB **전부 체크**.

### 5. 전체 파이프라인 (Windows)

**PowerShell** — `machine-learning-project`에서:

```powershell
cd machine-learning-project
pip install -e .

$env:KMA_APIHUB_AUTH_KEY = "your_key"

python scripts\data_collection\kbo_scraping.py
python scripts\data_collection\kbo_standings_scrape.py
python scripts\data_collection\kbo_size.py
python scripts\data_collection\weather_api.py
python scripts\preprocessing\preprocess_attendance_weather.py
python scripts\features\build_features.py
python scripts\eda\run_eda.py
python scripts\modeling\benchmark_models.py
python scripts\modeling\evaluate_model.py
python scripts\modeling\tune_hyperparams.py --n-trials 50
```

### 6. 가상환경 (Windows · 선택)

**PowerShell:**

```powershell
cd machine-learning-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
streamlit run scripts\app\streamlit_app.py
```

`running scripts is disabled` 오류 시 (한 번만):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**cmd:**

```cmd
cd machine-learning-project
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
streamlit run scripts\app\streamlit_app.py
```

### 7. Windows 문제 해결

| 증상 | 해결 |
|------|------|
| `pyproject.toml not found` | `machine-learning-project`로 `cd` 후 `pip install -e .` |
| `cd Machine-Learning-Project\...` 실패 | 이미 루트 안 → `cd machine-learning-project` 만 |
| `python` 인식 안 됨 | 설치 시 **Add to PATH** 또는 `py -3.12` |
| `Activate.ps1` 거부 | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `streamlit` 없음 | `python -m streamlit run scripts\app\streamlit_app.py` |
| EDA 차트 한글 □□□ | `run_eda.py`가 mac용 `AppleGothic` 사용 — `eda_summary.md` 텍스트는 정상 |
| Selenium 실패 | Chrome 설치, 백신이 chromedriver 차단 여부 확인 |

**루트에서 Cloud와 동일 설치 (선택)**

```powershell
cd Machine-Learning-Project
pip install -r requirements.txt
```

---

## 3개 모델 (RF · LGBM · XGB)

| 목적 | 스크립트 | 결과 |
|------|----------|------|
| **3개 한 번에** 학습·비교 | `benchmark_models.py` | RF + LGBM + XGB `.joblib`, `model_benchmark.json` |
| **RF만** | `train_model.py` | `attendance_rf_pipeline.joblib` |

- **3번 따로 실행할 필요 없음** — `benchmark_models.py` 한 번이면 RF → LGBM → XGB 순서로 저장.
- **사전 조건:** `data/processed/kbo_train_ready.csv` (없으면 `build_features.py` 또는 repo 포함 데이터).
- **XGB:** `pip install -e ".[benchmark]"` (xgboost 포함).

**Streamlit:** ML 알고리즘에서 여러 모델 체크 → 예측 **평균**. repo에 3개 joblib 있으면 학습 생략 가능.

### 벤치마크 (테스트 287경기)

| 지표 | Dummy | 구장평균 | RF | LightGBM | XGBoost |
|------|-------|----------|-----|----------|---------|
| MAE | 4,671 | 3,945 | 1,964 | **1,911** | 1,928 |
| R² | -0.03 | 0.33 | 0.74 | **0.77** | 0.77 |

---

## Streamlit Cloud

| 항목 | 값 |
|------|-----|
| Repository | `enzun123/Machine-Learning-Project` |
| Branch | `main` |
| Main file | `machine-learning-project/scripts/app/streamlit_app.py` |
| Python | 3.12 권장 |

- 루트 `requirements.txt`, `packages.txt` (chromium·한글 폰트 — Linux 전용)
- **Secrets:** 배포 관리 화면에서 `KMA_APIHUB_AUTH_KEY` 등록 (키는 repo에 커밋 금지)

| 기능 | 로컬 (Mac/Win) | Cloud |
|------|----------------|-------|
| RF/LGBM/XGB 예측 | ✅ | ✅ |
| 동네예보 | API 키 / secrets.toml | Secrets |
| 최근 5경기 크롤 | Chrome (기본 ON) | 불안정 (OFF 권장) |

---

## 모델 성능 · 스크립트 · 팀

### Streamlit UI

| 기능 | 설명 |
|------|------|
| 단일 경기 예측 | 날짜·구장·팀·기상 → 관중·혼잡도 |
| CSV 일괄 | `경기날짜, 홈팀, 방문팀, 구장` 업로드 |
| 대체 구장 | 포항·울산·청주 prior |
| 한계 | 2024–25 학습 — 2026·미래 일정 오차 가능 |

### 주요 스크립트

| 경로 | 역할 |
|------|------|
| `app/streamlit_app.py` | 메인 웹앱 |
| `app/csv_batch_predict_ui.py` | CSV 일괄 UI |
| `modeling/benchmark_models.py` | 3모델 |
| `modeling/train_model.py` | RF만 |
| `features/build_features.py` | 피처 |
| `eda/run_eda.py` | EDA |
| `data_collection/kbo_scraping.py` | 관중 크롤링 |
| `common/kma_vilage_fcst.py` | 동네예보 API |

### Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `main` | Cloud 배포 |
| `develop` | 기능 통합 |
| `feat/*` | 기능별 개발 |

### 팀 · 문의

| 역할 | 이름 |
|------|------|
| 팀장 | 허은준 (enzun123) — enzun123@gmail.com |
| 팀원 | 김지원, 이승민, 최종원 |

교육·팀 프로젝트 (KBO 관중 예측 ML).
