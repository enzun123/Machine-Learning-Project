# 🏟️ feat/stadium-capacity

## 📌 브랜치 목적
KBO 구장별 최대 수용인원 데이터를 정리하고, 모델링 전처리 과정에서 참조할 수 있는 마스터 데이터를 관리합니다.

## 🚀 바로 실행하기
아래 패키지를 설치한 후 구장 정보 생성 스크립트를 실행하세요.

```bash
# 1) 데이터 처리 필수 패키지 설치
pip install pandas

# 2) 구장 수용인원 데이터 생성 실행
python machine-learning-project/scripts/data_collection/kbo_size.py