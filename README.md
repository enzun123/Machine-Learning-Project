# KBO 관중 기록 수집 — GraphDaily 스크래핑 (`feat/scraping-kbo`)

**브랜치:** `feat/scraping-kbo`

KBO 기록실 **일자별 관중(GraphDaily)** 페이지를 Selenium으로 돌며 연도별 표를 페이지네이션 수집하고, 행 스키마(신형 헤더 / 구형 6열 등)에 맞춰 파싱한 뒤 **`enrich_attendance_df`** 로 학습·상류 파이프라인에 맞는 컬럼으로 정리해 **`data/raw/kbo_{연도}_attendance.csv`** 로 저장합니다.

## 브랜치 개요

- **수집 대상:** `https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx`  
- **CLI 출력:** 연도별 CSV — `machine-learning-project/data/raw/kbo_{year}_attendance.csv` (UTF-8 BOM)  
- **주요 컬럼:** `연도`, `경기날짜`, `경기시간`, `경기일시`, `요일`, `평일_주말`, `월`, `주차_ISO`, `홈팀`, `방문팀`, `구장`, `기상_매핑_지역키`, `관중수`  
- **기상 조인 힌트:** `기상_매핑_지역키`는 `weather_api` 등과 맞춘 구장→권역 문자열(`STADIUM_TO_REGION_KEY`)

## 전체 코드 실행 순서

1. **`feat/scraping-kbo`** → `data/raw/kbo_*_attendance.csv`  
2. `feat/weather-api` — 관측 기상과 병합해 interim `kbo_*_attendance_weather.csv` 등  
3. `feat/preprocessing` — `final_dataset.csv`  
4. `feat/stadium-capacity` … 이하 동일  

## 주요 구조

```text
machine-learning-project/
├── data/raw/
│   ├── kbo_2024_attendance.csv    # ★ 기본 산출 예
│   └── kbo_2025_attendance.csv
└── scripts/data_collection/
    └── kbo_scraping.py            # ★ CLI: scrape_kbo_attendance + enrich
```

다른 스크립트(`fetch_recent_crowd.py` 등)에서 **`scrape_kbo_attendance` / `enrich_attendance_df`** 를 import 해 재사용할 수 있습니다.

## 주요 기능

| 항목 | 설명 |
|------|------|
| `--years` | 수집 연도 목록 (**기본:** `2024 2025`) |
| `--headed` | 헤드리스 끄고 브라우저 표시(레이아웃·차단 디버그) |
| `KBO_SCRAPE_HEADLESS` | `0`이면 헤드리스 끔 (`--headed`와 동일 계열) |
| 표 호환 | GraphDaily **신형(홈/방문 열)**·**구형(시간·vs 텍스트)** 등 자동 스키마 감지 |
| 품질 로그 | 날짜 파싱 실패 행 제거, 관중수 이상(5만 초과·음수) **경고**, 동일 경기 키 중복 경고 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas beautifulsoup4 selenium webdriver-manager
```

Chrome(Chromium 계열)이 있어야 하며, `ChromeDriverManager`가 드라이버를 받습니다.

### 2. 스크래핑 실행

저장소 루트에서:

```bash
python3 machine-learning-project/scripts/data_collection/kbo_scraping.py
python3 machine-learning-project/scripts/data_collection/kbo_scraping.py --years 2023 2024 --headed
```

또는:

```bash
cd machine-learning-project
python3 scripts/data_collection/kbo_scraping.py --years 2025
```

**모든 연도가 비면** exit code `1`과 함께 `--headed`로 사이트·차단 여부를 확인하라는 메시지가 stderr에 출력됩니다.

### 3. (선택) `PYTHONPATH`

다른 디렉터리에서 모듈만 import 할 때는 저장소 루트에서:

```bash
export PYTHONPATH=machine-learning-project/scripts
```

```python
from data_collection.kbo_scraping import scrape_kbo_attendance, enrich_attendance_df, scraped_rows_to_dataframe
```

## 데이터 흐름

1. 연도별로 시즌 드롭다운 선택 → 검색/조회 클릭  
2. 표 HTML 파싱 → 페이지 `next` 클릭으로 전 페이지 순회  
3. 행을 dict 또는 list로 적재 → `scraped_rows_to_dataframe`  
4. `enrich_attendance_df`에서 홈/방문 분리, `경기일시`·`월`·`주차_ISO` 파생, 관중수 정수화·결측 제거  
5. `data/raw/kbo_{year}_attendance.csv` 저장  

## 주의 사항

- **사이트 구조·로봇 정책 변경** 시 스크립트 수정이 필요할 수 있습니다.  
- **관중수 이상치:** 현재는 **로그 경고만** 하고 행을 임의로 자르지 않습니다. 전처리·피처 단계와 합쳐 **팀 정책(제거·클리핑·유지)** 을 정해 두면 혼선이 줄어듭니다.  
- **수집 빈도:** KBO 서버 부하를 고려해 페이지 간 `sleep`이 들어 있습니다. 무리한 병렬화는 피하세요.

## 관련 브랜치·모듈

- **`feat/weather-api`:** `기상_매핑_지역키`로 기상 CSV와 조인  
- **`scripts/data_collection/fetch_recent_crowd.py`:** 동일 스크랩 함수로 **구장별 최근 N경기**만 추출  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
