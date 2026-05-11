# KBO 관중 예측 ML 프로젝트 (`develop`)

## 전체 코드 실행 순서

1. `feat/scraping-kbo`
2. `feat/weather-api`
3. `feat/preprocessing`
4. `feat/stadium-capacity` (구장 정본 `kbo_stadium_info.csv`)
5. `feat/eda`
6. `feat/feature-engineering` (`build_features.py` → `kbo_train_ready.csv`)
7. `feat/ml-modeling`
8. `feat/streamlit-ui`

## `feat/eda` (탐색만, 모델 입력 스키마는 변경하지 않음)

- **입력:** `machine-learning-project/data/processed/final_dataset.csv` (및 EDA용 `kbo_stadium_info.csv` 조인)
- **산출:** `machine-learning-project/reports/eda/` 요약·그림
- **`kbo_train_ready.csv`·학습용 컬럼 확정**은 `feat/feature-engineering`에서 처리합니다.

```bash
pip install pandas matplotlib seaborn
python3 machine-learning-project/scripts/eda/run_eda.py
```

## `feat/ml-modeling`

`kbo_train_ready.csv`로 베이스라인 학습·평가(joblib 파이프라인, 리포트)를 수행합니다.

```bash
pip install pandas numpy scikit-learn joblib
python3 machine-learning-project/scripts/modeling/train_model.py
python3 machine-learning-project/scripts/modeling/evaluate_model.py
```
