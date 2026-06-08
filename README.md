# Optuna 하이퍼파라미터 튜닝 (`feat/model-hyperparameter-tuning`)

**브랜치:** `feat/model-hyperparameter-tuning`

**Optuna + TimeSeriesSplit**으로 RF / LightGBM / XGBoost 하이퍼파라미터를 탐색하고, 최적 파라미터를 JSON에 저장한 뒤 **joblib·벤치마크 지표**를 갱신하는 브랜치입니다.

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **스크립트** | `scripts/modeling/tune_hyperparams.py` |
| **탐색** | Optuna TPE (베이지안), `TimeSeriesSplit(n_splits=5)` — 시간순 누수 방지 |
| **스코어** | `neg_MAE` (낮을수록 좋음) |
| **출력 JSON** | `best_params.json` (RF), `best_lgbm_params.json`, `best_xgb_params.json` |
| **후속** | `train_model.py` / `benchmark_models.py`가 JSON을 읽어 재학습 |

---

## 변경·갱신 파일

| 파일 | 내용 |
|------|------|
| `scripts/modeling/tune_hyperparams.py` | RF / LGBM / XGB Optuna 튜닝 CLI |
| `scripts/modeling/train_model.py` | `best_params.json` 연동 |
| `scripts/modeling/benchmark_models.py` | LGBM·XGB best params 연동 |
| `models/best_*.json` | 튜닝된 하이퍼파라미터 |
| `models/attendance_*_pipeline.joblib` | 튜닝 반영 재학습 모델 |
| `reports/modeling/model_benchmark.json` | 테스트셋 벤치마크 지표 |

---

## 벤치마크 (테스트 359경기, `365d9c8` 기준)

| 지표 | Dummy | 구장평균 | RF | LightGBM | XGBoost |
|------|-------|----------|-----|----------|---------|
| MAE | 4,658 | 3,979 | 1,958 | **1,856** | 1,884 |
| R² | -0.04 | 0.32 | 0.75 | **0.78** | 0.77 |

**best_by_mae:** LightGBM

---

## 실행 방법

### 1. 의존성 (Optuna)

```bash
cd machine-learning-project
pip install -e ".[benchmark]"
```

(`optuna`는 `[benchmark]` extra에 포함. `pip install -e .`만으로는 RF/LGBM/XGB 학습 가능, 튜닝은 extra 필요.)

### 2. 튜닝

```bash
cd machine-learning-project
python3 scripts/modeling/tune_hyperparams.py --model all --n-trials 50
python3 scripts/modeling/tune_hyperparams.py --model lgbm --n-trials 30
python3 scripts/modeling/tune_hyperparams.py --model rf
```

| 옵션 | 설명 |
|------|------|
| `--model` | `rf` · `lgbm` · `xgb` · `all` |
| `--n-trials` | Optuna 시도 횟수 (기본 30) |

### 3. 튜닝 후 재학습·벤치마크

```bash
python3 scripts/modeling/benchmark_models.py   # 3모델 한 번에
python3 scripts/modeling/train_model.py          # RF만
```

**사전 조건:** `data/processed/kbo_train_ready.csv` (없으면 `build_features.py` 실행)

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `1440128` | `tune_hyperparams.py` 도입 (패키지 레이아웃과 함께) |
| `365d9c8` | 튜닝 결과 반영, joblib·`model_benchmark.json` 갱신 (**브랜치 tip**) |

---

## 상위 브랜치

- **`feat/ml-modeling`** — RF 파이프라인·`train_model.py`  
- **`feat/model-benchmark`** — RF/LGBM/XGB `benchmark_models.py`  
- **`feat/model-hyperparameter-tuning`** — Optuna 튜닝·best params (**현재**)
