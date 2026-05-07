# 🌤️ feat/weather-api

## 📌 브랜치 목적
기상청 ASOS API를 통해 구장별 날씨 데이터를 호출하고 정제합니다.

## 🚀 전체 코드 실행순서
1. `feat/scraping-kbo`
2. `feat/weather-api`
3. `feat/preprocessing`
4. `feat/eda`
5. `feat/feature-engineering`
6. `feat/streamlit-ui`

## 🚀 바로 실행하기
아래 패키지를 설치한 후 API 호출 스크립트를 실행하세요.

```bash
# 1) API 통신 및 데이터 처리 패키지 설치
pip install pandas requests

# 2) 기상 데이터 호출 실행
python machine-learning-project/scripts/data_collection/weather_api.py