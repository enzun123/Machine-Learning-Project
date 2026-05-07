# 🛠️ feat/feature-engineering

## 📌 브랜치 목적
EDA 인사이트를 바탕으로 학습용 최종 데이터셋(`kbo_train_ready.csv`)을 생성합니다.

## 🚀 전체 코드 실행순서
1. `feat/scraping-kbo`
2. `feat/weather-api`
3. `feat/preprocessing`
4. `feat/eda`
5. `feat/feature-engineering`
6. `feat/streamlit-ui`

## 🚀 바로 실행하기
아래 패키지를 설치한 후 피처 엔지니어링 스크립트를 실행하세요.

```bash
# 1) 필수 패키지 설치
pip install pandas numpy

# 2) 실행 (README.md가 있는 루트 기준)
python machine-learning-project/scripts/features/build_features.py