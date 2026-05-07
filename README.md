# 🏟️ feat/scraping-kbo

## 📌 브랜치 목적
KBO 공식 홈페이지에서 경기 일정 및 관중수 데이터를 수집합니다.

## 🚀 바로 실행하기
아래 패키지를 설치한 후 크롤링 스크립트를 실행하세요.

```bash
# 1) 크롤링 필수 패키지 설치
pip install pandas selenium beautifulsoup4 requests webdriver-manager

# 2) 크롤링 스크립트 실행
python machine-learning-project/scripts/data_collection/kbo_scraping.py