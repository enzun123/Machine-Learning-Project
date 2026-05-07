# 🗂️ feat/preprocessing

## 📌 브랜치 목적
KBO 경기 데이터와 기상 API 데이터를 날짜 및 구장 기준으로 병합하고 기초 결측치를 정제합니다.

## 🚀 전체 코드 실행순서
1. `feat/scraping-kbo`
2. `feat/weather-api`
3. `feat/preprocessing`
4. `feat/eda`
5. `feat/feature-engineering`

## 🚀 바로 실행하기
아래 패키지를 설치한 후 데이터 병합 스크립트를 실행하세요.

```bash
# 1) 전처리 필수 패키지 설치
pip install pandas numpy

# 2) 통합 데이터셋 생성 스크립트 실행
python machine-learning-project/scripts/preprocessing/preprocess_attendance_weather.py
```
