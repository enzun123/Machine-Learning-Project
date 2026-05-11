# 📊 feat/eda

## 📌 브랜치 목적
전처리 산출물인 `final_dataset.csv`를 기준으로 탐색·시각화하여 관중수에 영향을 줄 만한 요인과 **피처 설계 아이디어**를 정리합니다.

- **입력:** `machine-learning-project/data/processed/final_dataset.csv` (및 EDA용 `kbo_stadium_info.csv` 조인)
- **산출:** `reports/eda/` 요약·그림 — **`kbo_train_ready.csv`나 모델 입력 스키마는 이 브랜치에서 바꾸지 않습니다.** 확정 피처는 `feat/feature-engineering`의 `build_features.py`에서 반영합니다.

## 🚀 전체 코드 실행순서
1. `feat/scraping-kbo`
2. `feat/weather-api`
3. `feat/preprocessing`
4. `feat/eda`
5. `feat/feature-engineering`
6. `feat/streamlit-ui`

## 🚀 바로 실행하기
아래 패키지를 설치한 후 시각화 스크립트를 실행하세요.

```bash
# 1) 시각화 및 데이터 분석 패키지 설치
pip install pandas matplotlib seaborn

# 2) EDA 리포트 및 그래프 생성
python machine-learning-project/scripts/eda/run_eda.py