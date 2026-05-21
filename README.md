# ⚾ KBO 관중 예측 ML 프로젝트

> KBO 경기별 관중 수를 머신러닝으로 예측하는 엔드투엔드 파이프라인 + Streamlit 웹앱

**🌐 데모:** [kbo-ml-prediction.streamlit.app](https://kbo-ml-prediction.streamlit.app)

---

## 📌 프로젝트 개요

경기 일정·구장·기상·순위를 결합해 관중 수를 예측합니다. **Streamlit**에서 예측·혼잡도·동네예보(우천 참고)를 확인할 수 있습니다.

- 피처: 승률·페넌트·매치업·최근 5경기·시즌 진행률 등
- 모델: RandomForest(기본) + LightGBM + XGBoost (선택)
- 데이터·학습된 모델이 repo에 있으면 **수집·학습 없이 앱만** 실행 가능

### 📦 저장소에 포함된 것 (앱만 켤 때)

| 경로 (`machine-learning-project/` 기준) | 설명 |
|----------------------------------------|------|
| `data/processed/kbo_train_ready.csv` | 학습용 피처 |
| `models/attendance_rf_pipeline.joblib` | RF |
| `models/attendance_lgbm_pipeline.joblib` | LGBM |
| `models/attendance_xgb_pipeline.joblib` | XGB |

---

## 📁 폴더 구조 (중요)

```
Machine-Learning-Project/          ← git clone 한 루트 (README 여기)
├── requirements.txt               ← Streamlit Cloud용
└── machine-learning-project/      ← ★ pip·streamlit·pytest는 여기서!
    ├── pyproject.toml               ← pip install -e . 대상
    ├── data/
    ├── models/
    └── scripts/
        └── app/streamlit_app.py
```

> **주의:** `pyproject.toml`은 **루트가 아니라** `machine-learning-project/` 안에 있습니다.  
> 루트에서 `pip install -e .` 하면 `setup.py / pyproject.toml not found` 오류가 납니다.

---

## ⚡ 빠른 시작 (가상환경 없음 · 학교 PC 권장)

**Python 3.10~3.12** 권장 (3.14 등 최신 버전은 패키지 호환 문제가 날 수 있음).

### 1) 설치 (한 번만)

**macOS**

```bash
git clone https://github.com/enzun123/Machine-Learning-Project.git
cd Machine-Learning-Project/machine-learning-project

pip3 install -U pip
pip3 install -e .
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/enzun123/Machine-Learning-Project.git
cd Machine-Learning-Project\machine-learning-project

python -m pip install -U pip
pip install -e .
```

| 명령 | 설명 |
|------|------|
| `pip install -e .` | 현재 폴더(`machine-learning-project`) 패키지를 **전역(또는 사용자) Python**에 설치 |
| 가상환경 `.venv` | **필수 아님** — 다른 ML 과제와 패키지가 꼬이면 [선택: 가상환경](#-선택-가상환경-venv) 사용 |

### 2) Streamlit 실행

**macOS**

```bash
cd Machine-Learning-Project/machine-learning-project
streamlit run scripts/app/streamlit_app.py
```

**Windows**

```powershell
cd Machine-Learning-Project\machine-learning-project
streamlit run scripts\app\streamlit_app.py
```

브라우저: `http://localhost:8501`  
`streamlit` 명령이 없으면: `python -m streamlit run scripts/app/streamlit_app.py`

### 3) (선택) 테스트

```bash
cd machine-learning-project
pip install -e ".[dev]"
pytest
```

---

## 🤖 3개 모델 (RF · LGBM · XGB)

**따로 3번 학습할 필요 없음.** `benchmark_models.py` **한 번**이면 3개 `.joblib` 생성.

```bash
cd machine-learning-project
pip install -e ".[benchmark]"
python scripts/modeling/benchmark_models.py    # Windows: python, macOS: python3
```

**Streamlit:** 사이드바 **ML 알고리즘**에서 RF · LGBM · XGB **전부 체크** → 예측은 **평균**.  
repo에 3개 joblib이 이미 있으면 **학습 생략**하고 체크만 하면 됨.

| 목적 | 스크립트 |
|------|----------|
| RF만 | `train_model.py` |
| 3개 + 비교표 | `benchmark_models.py` |

---

## 🔄 전체 파이프라인 (처음부터 돌릴 때)

`machine-learning-project` 폴더에서 실행. `pip install -e .` 후에는 **`PYTHONPATH` 보통 불필요**.

| 단계 | 스크립트 | 비고 |
|------|----------|------|
| 1 | `kbo_scraping.py`, `kbo_standings_scrape.py` | Chrome + Selenium |
| 2 | `kbo_size.py` | 구장 정원 |
| 3 | `weather_api.py` | `KMA_APIHUB_AUTH_KEY` |
| 4 | `preprocess_attendance_weather.py` | |
| 5 | `build_features.py` | |
| 6 | `run_eda.py` | 선택 (Windows 차트 한글: mac 전용 폰트 이슈 있음) |
| 7 | `train_model.py` 또는 `benchmark_models.py` | |
| 8 | `evaluate_model.py` | 선택 (RF) |

**macOS 예시**

```bash
cd machine-learning-project
export KMA_APIHUB_AUTH_KEY="your_key"   # 3단계만
python3 scripts/data_collection/kbo_scraping.py
# ... 이하 동일 패턴
```

**Windows (PowerShell) 예시**

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

## 🔑 기상 API · 선택 설정

| 변수 | 용도 |
|------|------|
| `KMA_APIHUB_AUTH_KEY` | 동네예보·`weather_api.py` |
| `STREAMLIT_WEB_RECENT=0` | 최근 5경기 크롤 끔 |

| OS | 예 |
|----|-----|
| macOS | `export KMA_APIHUB_AUTH_KEY="키"` |
| Windows | `$env:KMA_APIHUB_AUTH_KEY = "키"` |

로컬: `machine-learning-project/.streamlit/secrets.toml` 에 `KMA_APIHUB_AUTH_KEY` (선택)  
Cloud: 앱 **Settings → Secrets** (배포 관리자만 설정)

---

## 📦 선택: 가상환경 (venv)

다른 프로젝트와 **pandas / scikit-learn 버전이 섞일 때**만 사용.

**macOS**

```bash
cd machine-learning-project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

**Windows**

```powershell
cd machine-learning-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
streamlit run scripts\app\streamlit_app.py
```

`Activate.ps1` 오류 시: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## 🪟 Windows · macOS 문제 해결

| 증상 | 해결 |
|------|------|
| `pyproject.toml not found` | **`machine-learning-project`로 cd** 후 `pip install -e .` |
| `cd Machine-Learning-Project/...` 실패 | 이미 clone 루트에 있으면 `cd machine-learning-project` 만 |
| `python` 없음 | `py -3.12` 사용 |
| `streamlit` 없음 | `python -m streamlit run ...` |
| EDA 차트 한글 □□□ (Windows) | `run_eda.py`가 `AppleGothic`(맥 전용) 사용 — md 요약은 정상 |

**루트에서 Cloud와 동일 설치 (선택)**

```bash
cd Machine-Learning-Project    # clone 루트
pip install -r requirements.txt
```

---

## ☁️ Streamlit Cloud

| 항목 | 값 |
|------|-----|
| Main file | `machine-learning-project/scripts/app/streamlit_app.py` |
| Branch | `main` |
| Python | 3.12 권장 |

루트 `requirements.txt`, `packages.txt` 자동 사용. Secrets에 `KMA_APIHUB_AUTH_KEY` (선택).

---

## ⚙️ 모델 요약

- **1,436경기**, 시간 순 홀드아웃 (테스트 2025.7~10)
- 벤치마크 MAE: Dummy 4,671 → RF **1,964** → LGBM **1,911** → XGB 1,928

| 지표 | RF | LightGBM | XGBoost |
|------|-----|----------|---------|
| MAE | 1,964 | **1,911** | 1,928 |
| R² | 0.74 | **0.77** | 0.77 |

---

## 📊 주요 스크립트

| 경로 | 역할 |
|------|------|
| `app/streamlit_app.py` | 웹앱 |
| `app/csv_batch_predict_ui.py` | CSV 일괄 예측 |
| `modeling/benchmark_models.py` | 3모델 학습·비교 |
| `modeling/train_model.py` | RF만 |
| `features/build_features.py` | 피처 |
| `eda/run_eda.py` | EDA (그림·`eda_summary.md`) |
| `common/*` | 구장·기상·혼잡도 |

---

## 🌿 Git 브랜치

| 브랜치 | 용도 |
|--------|------|
| `main` | Cloud 배포 |
| `develop` | 기능 통합 |
| `feat/*` | 기능별 히스토리 |

---

## 👥 팀 · 문의

| 역할 | 이름 |
|------|------|
| 팀장 | 허은준 (enzun123) — enzun123@gmail.com |
| 팀원 | 김지원, 이승민, 최종원 |

교육·팀 프로젝트 (KBO 관중 예측 ML).
