# 3모델 벤치마크 — RF · LGBM · XGB (`feat/model-benchmark`)

**브랜치:** `feat/model-benchmark`

동일 **시간순 train/test 분할**·동일 피처로 **RandomForest · LightGBM · XGBoost**를 한 번에 학습·비교하고, **joblib 3개**와 **`model_benchmark.json`** 을 저장하는 브랜치입니다. Streamlit에서 ML 알고리즘 다중 선택·평균 예측의 기반이 됩니다.

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **스크립트** | `scripts/modeling/benchmark_models.py` |
| **입력** | `data/processed/kbo_train_ready.csv` |
| **분할** | `temporal_연도_월_주차_ISO` (미래 누수 방지) |
| **베이스라인** | Dummy(전체 평균), 구장별 평균 |
| **출력 joblib** | `attendance_rf_pipeline.joblib`, `attendance_lgbm_pipeline.joblib`, `attendance_xgb_pipeline.joblib` |
| **출력 리포트** | `reports/modeling/model_benchmark.json` |

---

## 변경·갱신 파일

| 파일 | 내용 |
|------|------|
| `scripts/modeling/benchmark_models.py` | 3모델 학습·지표·저장 CLI (**신규**) |
| `scripts/modeling/tune_hyperparams.py` | LGBM·XGB 파이프라인 빌더 (`best_*_params.json` 연동) |
| `scripts/app/streamlit_app.py` | RF/LGBM/XGB joblib 로드·다중 선택 평균 예측 |
| `scripts/features/build_features.py` | 구장 정원·KIA 상한 등 (벤치마크 데이터 정합) |
| `models/attendance_*_pipeline.joblib` | 3종 파이프라인 |
| `reports/modeling/model_benchmark.json` | MAE·R²·best_by_mae |

---

## 벤치마크 (테스트 287경기, `b849b60` 기준)

| 지표 | Dummy | 구장평균 | RF | LightGBM | XGBoost |
|------|-------|----------|-----|----------|---------|
| MAE | 4,674 | 3,952 | **1,983** | 2,058 | 2,018 |
| R² | -0.03 | 0.33 | 0.72 | 0.73 | **0.75** |

**best_by_mae:** RandomForest

> 이후 `feat/model-hyperparameter-tuning`에서 Optuna 튜닝 후 LGBM MAE **1,856** 등으로 지표가 갱신됩니다.

---

## 실행 방법

### 1. 설치

```bash
cd machine-learning-project
pip install -e .
```

(XGBoost는 `pyproject.toml` 기본 포함. Optuna 튜닝만 `pip install -e ".[benchmark]"`.)

### 2. 벤치마크 (3모델 한 번에)

```bash
cd machine-learning-project
python3 scripts/modeling/benchmark_models.py
```

**사전 조건:** `data/processed/kbo_train_ready.csv`  
(없으면 `python3 scripts/features/build_features.py`)

### 3. Streamlit에서 확인

```bash
streamlit run scripts/app/streamlit_app.py
```

사이드바 **ML 알고리즘** → RF · LGBM · XGB **전부 체크** → 예측 **평균**.

---

## `train_model.py` 와 차이

| 스크립트 | 결과 |
|----------|------|
| `train_model.py` | RF `.joblib` **1개**만 |
| `benchmark_models.py` | RF + LGBM + XGB **3개** + `model_benchmark.json` |

**3번 따로 실행할 필요 없음** — `benchmark_models.py` 한 번이면 됩니다.

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `aeab23a` | `benchmark_models.py` 도입, LGBM·XGB joblib·벤치마크 JSON·Streamlit 다중 모델 |
| `b849b60` | 벤치마크 재실행, 3종 joblib 갱신 (**브랜치 tip**) |

---

## 상위·후속 브랜치

- **`feat/ml-modeling`** — RF `train_model.py`  
- **`feat/model-benchmark`** — 3모델 비교·저장 (**현재**)  
- **`feat/model-hyperparameter-tuning`** — Optuna → best params → 벤치마크 재갱신
