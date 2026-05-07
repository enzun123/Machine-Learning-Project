# ⚾ KBO 관람 수요 예측 프로젝트

## 📌 개요
기상 데이터와 경기 데이터를 결합해 관중 수요를 예측하고, 운영 최적화 인사이트를 제공하는 프로젝트입니다.

## 🚀 전체 코드 실행순서
1. `feat/scraping-kbo`
2. `feat/weather-api`
3. `feat/preprocessing`
4. `feat/eda`
5. `feat/feature-engineering`

## 📦 기본 패키지 설치
```bash
pip install pandas numpy matplotlib seaborn requests selenium beautifulsoup4 webdriver-manager
```

## ▶ 대표 실행 명령
```bash
# 관중 데이터 스크래핑
python machine-learning-project/scripts/data_collection/kbo_scraping.py

# 기상 데이터 연동
python machine-learning-project/scripts/data_collection/weather_api.py

# 전처리
python machine-learning-project/scripts/preprocessing/preprocess_attendance_weather.py

# EDA
python machine-learning-project/scripts/eda/run_eda.py

# 피처 엔지니어링
python machine-learning-project/scripts/features/build_features.py
```
