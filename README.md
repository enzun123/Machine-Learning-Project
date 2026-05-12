# KBO 관중 예측 — 모델 학습·평가 (`feat/ml-modeling`)

**브랜치:** `feat/ml-modeling`

`kbo_train_ready.csv`를 읽어 **시간 순 홀드아웃**으로 학습·검증하고, **Random Forest 회귀 파이프라인**(타깃 `log1p` 변환)을 `joblib`으로 저장합니다. 이어서 같은 테스트 구간으로 **재평가·잔차 분석·permutation importance**를 수행해 `eval_report.json`을 만듭니다.

## 브랜치 개요

- **입력:** `machine-learning-project/data/processed/kbo_train_ready.csv` (스크립트가 요구하는 수치·범주 피처·`관중수`·시간 키 `연도`/`월`/`주차_ISO` 포함)
- **처리 (1단계 `train_model.py`):** 시간 순 분할 → 더미·구장 평균 베이스라인과 비교 → RF 파이프라인 학습 → 메트릭·분할 정보를 `train_report.json`에 기록
- **처리 (2단계 `evaluate_model.py`):** 저장된 파이프라인·`test_indices.npy`로 테스트 행만 다시 예측 → 구장별 MAE, permutation importance 상위 등 → `eval_report.json`
- **산출 (주요 파일):**
  - `machine-learning-project/models/attendance_rf_pipeline.joblib`
  - `machine-learning-project/models/test_indices.npy`
  - `machine-learning-project/models/train_report.json`
  - `machine-learning-project/models/eval_report.json`

## 전체 코드 실행 순서

1. `feat/scraping-kbo`  
2. `feat/weather-api`  
3. `feat/preprocessing`  
4. `feat/stadium-capacity` — 구장 정본 `kbo_stadium_info.csv`  
5. `feat/eda`  
6. `feat/feature-engineering` — `build_features.py` → **`kbo_train_ready.csv`**  
7. **`feat/ml-modeling`**
8. `feat/streamlit-ui`

## 주요 구조

```text
machine-learning-project/
├── data/processed/
│   └── kbo_train_ready.csv       # 필수 입력 (이전 단계 산출)
├── scripts/modeling/
│   ├── train_model.py            # ★ 1단계 학습
│   └── evaluate_model.py       # ★ 2단계 평가
└── models/
    ├── attendance_rf_pipeline.joblib
    ├── test_indices.npy
    ├── train_report.json
    ├── eval_report.json
    └── best_params.json          # 선택 — 있으면 RF 하이퍼파라미터로 병합 적용
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| 시간 순 분할 | `연도` → `월` → `주차_ISO` 정렬 후 뒤쪽 **TEST_SIZE(기본 0.2)** 비율을 테스트로 사용 |
| 베이스라인 | 전역 평균(`DummyRegressor`)·구장별 평균 관중과 테스트 MAE/RMSE/R² 비교 |
| 본 모델 | `ColumnTransformer`(범주 One-Hot + 수치 통과) + `RandomForestRegressor`, 타깃 `log1p` / 역변환 `expm1` |
| 튜닝 연동 | `models/best_params.json`이 있으면 `n_estimators`, `max_depth` 등이 학습 시 반영 |
| 평가 스크립트 | 동일 테스트 인덱스로 재예측, permutation importance(기본 `n_repeats=10`), 구장별 MAE 상위 요약 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas numpy scikit-learn joblib
```

`evaluate_model.py`의 permutation importance가 **전 코어**를 쓰도록 `n_jobs=-1`로 되어 있어, 환경에 따라 실행이 무거울 수 있습니다.

### 2. 사전 조건

- `kbo_train_ready.csv`가 없으면 `train_model.py`가 종료하며 `build_features.py` 실행을 안내합니다.  
  (`feat/feature-engineering` 단계에서 생성)

### 3. 학습 → 평가 (저장소 루트 기준)

`Machine-Learning-Project` 루트에서:

```bash
python3 machine-learning-project/scripts/modeling/train_model.py
python3 machine-learning-project/scripts/modeling/evaluate_model.py
```

또는 `machine-learning-project` 디렉터리로 이동한 뒤:

```bash
cd machine-learning-project
python3 scripts/modeling/train_model.py
python3 scripts/modeling/evaluate_model.py
```

**순서:** 반드시 **`train_model.py`를 먼저** 실행해야 `evaluate_model.py`가 요구하는 `attendance_rf_pipeline.joblib`와 `test_indices.npy`가 생깁니다.

### 4. Python에서 모듈만 재사용할 때

다른 스크립트에서 `train_model`의 상수·로더를 쓰려면, `scripts/modeling`과 동일한 방식으로 경로를 잡을 수 있습니다.

```bash
export PYTHONPATH=machine-learning-project/scripts/modeling
```

(저장소 **루트**에서 실행. 현재 터미널 세션에만 적용되며, 영구 설정은 셸 설정 파일에 동일 줄을 추가.)

```python
from pathlib import Path
from train_model import load_training_table

csv_path = Path("machine-learning-project/data/processed/kbo_train_ready.csv")
df = load_training_table(csv_path)
```

일반적으로는 위 **CLI 두 줄**만으로 충분합니다.

## 데이터 흐름

1. `kbo_train_ready.csv` 로드 후 필수 컬럼 검사·결측 처리  
2. 시간 키로 정렬해 마지막 구간을 테스트 인덱스로 고정·저장  
3. 학습 구간으로 파이프라인 `fit` → `joblib` 저장  
4. 평가 스크립트가 동일 `X_test`, `y_test`로 메트릭·중요도·구장별 오차 집계 → JSON 저장  

## 주의 사항

- **scikit-learn / joblib 버전:** 학습 환경과 배포(예: Streamlit) 환경의 `scikit-learn`이 크게 다르면 `joblib` 로드 실패가 날 수 있습니다.  
- **`best_params.json`:** 형식이 맞지 않으면 학습이 실패할 수 있으니, 튜닝 파이프라인과 동일 스키마로만 두세요.  
- **타깃:** `관중수`는 `log1p` 변환 후 학습합니다. 해석 시 역변환된 스케일을 기준으로 합니다.

## 관련 브랜치

- **`feat/feature-engineering`:** `kbo_train_ready.csv` 생성  
- **`feat/eda`:** 스키마 확정 전 탐색 (`reports/eda/`)  
- **`feat/streamlit-ui`:** 학습 산출 `attendance_rf_pipeline.joblib` 소비  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
