# ⚾ KBO 관람 수요 예측 ML

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![CI](https://img.shields.io/badge/CI-pytest-blue?logo=githubactions&logoColor=white)

**LightGBM 기반 KBO 관중 예측 — MAE 1,856명, R² 0.78**

**🌐 라이브 데모:** [kbo-ml-prediction.streamlit.app](https://kbo-ml-prediction.streamlit.app)

**최종 업데이트:** 2026년 6월 · 학습 데이터 기준 **2024–2025 시즌**

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [설치](#설치)
- [Quick Start](#quick-start)
- [환경 설정 (선택)](#환경-설정-선택)
- [모델 성능](#모델-성능)
- [데이터](#데이터)
- [프로젝트 구조](#프로젝트-구조)
- [모델링](#모델링)
- [Streamlit 앱](#streamlit-앱)
- [한계점](#한계점-limitations)
- [CI](#ci)
- [팀](#팀)
- [참고 리포트](#참고-리포트)

---

## 프로젝트 개요

KBO 경기별 관중 수를 머신러닝으로 예측하고, Streamlit 웹앱으로 **혼잡도·우천 참고 정보**까지 제공하는 End-to-End ML 프로젝트입니다.

구장 운영·마케팅·입장 planning에서 “이번 경기 관중이 얼마나 올까?”는 핵심 질문입니다.  
본 프로젝트는 **경기 일정, 구장, 기상, 팀·매치업 이력**을 입력으로 관중 수를 회귀 예측하고, 결과를 **누구나 쓸 수 있는 웹 UI**로 제공합니다.

**만든 이유 (Motivation)**

- KBO 관중은 구장·요일·날씨·팀 인기·매치업에 따라 크게 달라져, 단순 평균으로는 오차가 큼
- 데이터 수집 → 전처리 → 피처 → 학습 → 배포까지 **End-to-End ML 파이프라인**을 한 repo에서 재현 가능하게 구성
- 발표·실무에서 바로 쓸 수 있도록 **Streamlit Cloud 배포** 및 CSV 일괄 예측 지원

---

## 설치

```bash
git clone https://github.com/enzun123/Machine-Learning-Project.git
cd Machine-Learning-Project/machine-learning-project

pip install -U pip
pip install -e .
```

Streamlit Cloud와 동일하게 루트에서 설치하려면:

```bash
cd Machine-Learning-Project
pip install -r requirements.txt
```

| 항목 | 내용 |
|------|------|
| Python | **3.10 ~ 3.12** (`.python-version`: 3.12) |
| Chrome | KBO 크롤링·«최근 5경기» 사용 시 (로컬) |
| 기상 API (선택) | 동네예보 — [환경 설정](#환경-설정-선택) 참고 |

💡 **Tip:** repo에 `data/processed/kbo_train_ready.csv`와 `models/*.joblib`가 이미 포함되어 있으면, 크롤링·재학습 없이 **웹앱만** 바로 실행할 수 있습니다.

---

## Quick Start

### 웹앱 실행

```bash
cd machine-learning-project
streamlit run scripts/app/streamlit_app.py
```

브라우저: http://localhost:8501 · **Windows:** `python -m streamlit run scripts\app\streamlit_app.py`

### CSV 일괄 예측

사이드바 **예측 방식 → CSV 업로드 예측**에서 아래 형식의 CSV를 올립니다.

| 필수 컬럼 | 설명 |
|-----------|------|
| `경기날짜` | `YYYY-MM-DD` |
| `홈팀` | KBO 팀명 (예: 삼성, LG) |
| `방문팀` | KBO 팀명 |
| `구장` | 구장명 (예: 잠실, 대구) |

**샘플 파일 (repo 포함)**

- [일정 템플릿](machine-learning-project/data/external/batch_predict_schedule_template.csv) — 필수 4컬럼
- [기상 포함 템플릿](machine-learning-project/data/external/batch_predict_feature_template.csv) — 선택: 기온·강수·습도·풍속

앱 사이드바 **샘플 CSV** 버튼으로도 동일 파일을 내려받을 수 있습니다.

### 모델 재학습 (3종 한 번에)

```bash
cd machine-learning-project
python scripts/modeling/benchmark_models.py
```

→ `models/attendance_{rf,lgbm,xgb}_pipeline.joblib` + `reports/modeling/model_benchmark.json`

### 테스트

```bash
cd machine-learning-project
pip install -e ".[dev]"
pytest -q
```

---

## 환경 설정 (선택)

동네예보(우천 참고)를 쓰려면 [기상청 API허브](https://apihub.kma.go.kr/)에서 `KMA_APIHUB_AUTH_KEY`를 발급합니다. **키는 repo에 커밋하지 마세요.**

**로컬 — `machine-learning-project/.streamlit/secrets.toml` (권장)**

```toml
KMA_APIHUB_AUTH_KEY = "발급받은_키"
```

**로컬 — 셸 환경변수**

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_키"
```

Windows PowerShell: `$env:KMA_APIHUB_AUTH_KEY = "발급받은_키"`

**Streamlit Cloud** — 배포 관리 → **Secrets**에 동일 키 이름으로 등록 (형식은 위와 동일).

**기타 (로컬)**

```bash
export STREAMLIT_WEB_RECENT=0   # «최근 5경기 KBO 크롤» 끄기 (Cloud는 기본 OFF)
```

---

## 모델 성능

시계열 분할(`연도·월·ISO주차` 기준) 테스트셋 **359경기** (2025년 6–10월) · `model_benchmark.json` 기준

| 모델 | MAE (명) | R² | 비고 |
|------|----------|-----|------|
| Dummy (전체 평균) | 4,658 | -0.04 | 베이스라인 |
| 구장별 평균 | 3,979 | 0.32 | 베이스라인 |
| RandomForest | 1,958 | 0.75 | Optuna 튜닝 |
| **LightGBM** | **1,856** | **0.78** | **MAE 최우수** |
| XGBoost | 1,884 | 0.77 | Optuna 튜닝 |

- **평가 지표:** MAE (Mean Absolute Error), R²
- **최종 선택:** LightGBM — 비선형 관계·범주형 피처 처리에 유리하고, RF/XGB 대비 MAE가 가장 낮음
- **주요 피처 (Permutation Importance):** `matchup_prior_mean_att`, `weekday_sin`, `home_prior_mean_att`, `home_last5_mean_att`, `stadium_capacity`

구장별 오차는 편차가 큼 (예: 광주 MAE ~4,248명). KIA 홈 광주는 실제 판매 상한 **20,500명** 클립을 반영.

---

## 데이터

| 출처 | 내용 | 기간 |
|------|------|------|
| KBO 기록실 GraphDaily | 경기별 관중·일정·구장 | 2024–2025 |
| 기상청 API | 일별 기온·강수·습도·풍속 | 경기일 기준 |
| KBO 순위·구장 정보 | 승률·정원·대체 구장 | 2024–2025 |

**전처리 파이프라인 요약**

```
크롤/API 수집 → 결측·중복·구장명 통일 → 날씨 병합 (final_dataset.csv)
→ prior·날씨 bucket·승률·누수 방지 피처 (kbo_train_ready.csv)
→ 시계열 분할 학습 → joblib 저장 → Streamlit 추론
```

주요 정제: 강수 결측 0 처리, 더블헤더 정렬, 의심 관중 제거, KIA 광주 20,500명 상한 클립

---

## 프로젝트 구조

```
Machine-Learning-Project/              ← git clone 루트
├── requirements.txt                   ← Streamlit Cloud 의존성
├── packages.txt                       ← Linux: chromium, 한글 폰트
├── .python-version                    ← 3.12
└── machine-learning-project/          ← ★ 모든 명령은 여기서 실행
    ├── data/
    │   ├── raw/                       ← KBO 관중 원본 (2024–2025)
    │   ├── interim/                   ← 날씨 병합 중간 산출물
    │   ├── processed/                 ← final_dataset, kbo_train_ready
    │   └── external/                  ← 구장·순위·CSV 템플릿
    ├── models/                        ← *.joblib
    ├── reports/                       ← EDA·벤치마크·eval
    ├── tests/                         ← pytest
    └── scripts/
        ├── data_collection/           ← KBO·KMA 크롤/API
        ├── preprocessing/             ← 정제·병합
        ├── features/                  ← 피처 생성
        ├── modeling/                  ← 학습·튜닝·평가
        ├── eda/                       ← 탐색적 분석
        ├── app/                       ← Streamlit UI
        └── common/                    ← 설정·구장·기상 공통 모듈
```

---

## 모델링

| 단계 | 스크립트 | 설명 |
|------|----------|------|
| 피처 | `scripts/features/build_features.py` | EDA 기반 파생 변수, **타겟 누수 방지** (경기일 이전 데이터만) |
| 벤치마크 | `scripts/modeling/benchmark_models.py` | RF · LightGBM · XGBoost 비교·저장 |
| 튜닝 | `scripts/modeling/tune_hyperparams.py` | Optuna + TimeSeriesSplit, MAE 최소화 |
| 평가 | `scripts/modeling/evaluate_model.py` | 잔차·구장별 MAE·Permutation Importance |

**모델 선택 이유:** 트리 기반 앙상블(RF/LGBM/XGB)은 비선형·범주형(구장·팀·요일)에 강하고, 시계열 CV로 과적합을 줄이기 쉬움. 그중 **LightGBM**이 테스트 MAE·R² 모두 최우수.

Streamlit에서는 RF/LGBM/XGB를 **다중 선택 시 예측 평균**으로 표시.

---

## Streamlit 앱

| 기능 | 설명 |
|------|------|
| 단일 경기 예측 | 날짜·구장·팀·기온·강수·습도·풍속 → ML 예측·혼잡도·액션 플랜 |
| CSV 일괄 예측 | [샘플 CSV](#csv-일괄-예측) 업로드 → 일정별 예측 |
| 동네예보 | 기상청 API — 우천 취소 참고 ([환경 설정](#환경-설정-선택)) |
| 최근 5경기 차트 | 실제 vs 예측 비교 (로컬 CSV 또는 KBO 크롤) |

**Streamlit Cloud:** `enzun123/Machine-Learning-Project` · main file `machine-learning-project/scripts/app/streamlit_app.py`  
Cloud에서는 Selenium 크롤이 느리고 불안정하므로 **«최근 5경기 KBO 자동 반영» 기본 OFF** 권장.

---

## 한계점 (Limitations)

- **학습 구간:** 2024–2025 데이터 — 2026·미래 일정·신규 룰 변경 시 오차 증가 가능
- **구장별 편차:** 광주·울산 등 특정 구장 MAE가 크며, 소규모·대체 구장 데이터가 적음
- **실시간 요인 미반영:** 선발 투수·이벤트·프로모션·PO 긴장감 등은 모델에 직접 포함되지 않음
- **기상:** 경기 당일 실제 강우와 예보·과거 일별 기상 간 괴리 존재
- **크롤 의존:** KBO GraphDaily·Selenium 구조 변경 시 수집 스크립트 수정 필요

---

## CI

GitHub Actions (`.github/workflows/pytest.yml`): `main` / `develop` push·PR 시 Ubuntu + Python 3.11·3.12에서 `pytest -q` 실행.

---

## 팀

| 역할 | 이름 |
|------|------|
| 팀장 | 허은준 (enzun123) — enzun123@gmail.com |
| 팀원 | 김지원, 이승민, 최종원 |

---

## 참고 리포트

| 경로 | 내용 |
|------|------|
| `reports/modeling/model_benchmark.json` | 3모델 벤치마크 수치 |
| `models/eval_report.json` | RF 상세 평가·구장별 MAE |
| `reports/eda/eda_summary.md` | EDA 인사이트 |
| `reports/eda/feature_engineering_plan.txt` | 피처 설계·누수 방지 규칙 |
