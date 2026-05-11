🤖 feat/ml-modeling
📌 브랜치 목적
전처리 완료된 데이터(kbo_train_ready.csv)를 활용하여 KBO 관중수 예측 베이스라인 모델(Random Forest)을 학습시키고, UI 서비스에 연동할 최종 파이프라인 객체(attendance_rf_pipeline.joblib)와 성능 평가 리포트를 생성합니다.

🚀 바로 실행하기
아래 패키지를 설치한 후 머신러닝 학습 및 평가 스크립트를 실행하세요.

🚀 전체 코드 실행순서
feat/scraping-kbo

feat/weather-api

feat/preprocessing

feat/eda

feat/feature-engineering

feat/ml-modeling (현재 단계)

feat/streamlit-ui

Bash
# 1) 머신러닝 필수 패키지 설치
pip install pandas numpy scikit-learn joblib

# 2) 베이스라인 모델 학습 실행 (joblib 파이프라인 및 train_report 생성)
python machine-learning-project/scripts/modeling/train_model.py

# 3) (선택) 테스트 세트 상세 평가 및 피처 중요도(Feature Importance) 추출
python machine-learning-project/scripts/modeling/evaluate_model.py