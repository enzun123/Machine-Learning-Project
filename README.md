# KBO 구장별 최근 경기 관중 수집 (`feature/kbo-recent-games-scrape`)

**브랜치:** `feature/kbo-recent-games-scrape`

KBO 기록실 **GraphDaily** 페이지를 Selenium으로 스크랩한 뒤, 지정 **구장** 기준으로 **최근 N경기**의 관중 데이터만 골라 CSV로 저장하거나 표준 출력으로 확인하는 **CLI·라이브러리**입니다. Streamlit 등 다른 스크립트에서 `fetch_recent_games()`를 import 해 쓸 수 있습니다.

## 브랜치 개요

- **입력:** 구장명(`--stadium` / `-s`), 선택적으로 `--before` 기준일, `--years`, 경기 수 `--n`(기본 5)
- **처리:** `kbo_scraping.scrape_kbo_attendance`로 연도별 원시 수집 → `enrich_attendance_df`로 보강 → 구장 정규화(`common/stadium_aliases`) 후 날짜 역순으로 상위 N경기
- **산출:** 콘솔 표 또는 `-o`로 UTF-8 BOM CSV (`연도`, `경기날짜`, `홈팀`, `방문팀`, `구장`, `관중수` 등 컬럼이 있으면 우선 출력)

## 전체 코드 실행 순서

프로젝트 전체 파이프라인에서 이 브랜치는 **병렬로 붙는 수집 모듈**에 가깝습니다. 나머지 단계와의 대략적인 순서는 아래와 같습니다.

1. `feat/scraping-kbo` — KBO 관중 등 기본 수집 파이프라인  
2. `feat/weather-api`  
3. `feat/preprocessing`  
4. `feat/stadium-capacity` — `kbo_stadium_info.csv`  
5. `feat/eda`  
6. `feat/feature-engineering` — `kbo_train_ready.csv`  
7. `feat/ml-modeling`  
8. `feat/streamlit-ui` — (선택) UI에서 본 스크랩을 호출할 수 있음  

**이 브랜치:** `fetch_recent_crowd.py` / `fetch_recent_games()` — **구장별 최근 N경기**만 빠르게 뽑을 때 사용합니다. `kbo_scraping.py`가 있어야 동작합니다.

## 주요 구조

```text
machine-learning-project/
├── scripts/
│   ├── common/
│   │   └── stadium_aliases.py      # 구장명 정규화
│   └── data_collection/
│       ├── fetch_recent_crowd.py   # ★ 이 브랜치 핵심 CLI + fetch_recent_games()
│       └── kbo_scraping.py         # GraphDaily 스크랩·보강 (의존)
└── data/
    └── cache/                       # -o 저장 시 예: recent_창원.csv
```

## 주요 기능

| 기능 | 설명 |
|------|------|
| 구장 필터 | `STADIUM_ALIAS`로 표기 통일 후 해당 구장 경기만 선택 |
| `--n` | 가져올 최근 경기 개수 (**기본 5**) |
| `--before` | 해당 날짜 0시 **이전**에 치른 경기만 대상으로 최근 N경기 (예정 경기일 기준 직전 폼 확인용) |
| 연도 범위 | 기본 올해·전년, `--years`로 지정 가능 |
| 헤드리스 | 기본 헤드리스; `--headed` 또는 `KBO_SCRAPE_HEADLESS=0`으로 브라우저 표시 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas beautifulsoup4 selenium webdriver-manager
```

Chrome(또는 Chromium 계열)이 설치되어 있어야 합니다. `webdriver_manager`가 드라이버를 받습니다.

### 2. CLI 실행

저장소에서 **`machine-learning-project`** 디렉터리로 들어가 실행하는 것을 권장합니다.

```bash
cd machine-learning-project

python3 scripts/data_collection/fetch_recent_crowd.py -s 창원 --n 5
python3 scripts/data_collection/fetch_recent_crowd.py --stadium 대전 --before 2026-06-15 --n 5 -o data/cache/recent_대전.csv
python3 scripts/data_collection/fetch_recent_crowd.py -s 잠실 --years 2025 2024 --n 3
```

- `-s` / `--stadium`: **필수** (예: 창원, 대전, 잠실, 문학, 수원 …)  
- `--n`: 경기 수 (기본 5)  
- `-o` / `--output`: CSV 저장 경로 (없으면 표준 출력)  
- `--before YYYY-MM-DD`: 그날 0시 **미만** 경기만 보고 그중 최근 N경기  
- `--headed`: 브라우저 창 띄우기 (디버깅)

### 3. Python에서 호출

저장소 루트(`Machine-Learning-Project`)에서 `scripts` 패키지 루트를 잡아 주면 import 경로가 맞습니다.

```bash
export PYTHONPATH=machine-learning-project/scripts
```

```python
from data_collection.fetch_recent_crowd import fetch_recent_games

df = fetch_recent_games("창원", n=5, before="2026-06-15", headless=True)
```

위 `export`는 **현재 터미널 세션**에만 적용됩니다. 영구 설정은 셸 설정 파일(예: `~/.zshrc`)에 같은 줄을 넣으면 됩니다.

### 환경 변수

- `KBO_SCRAPE_HEADLESS` — `0`이면 헤드리스 끔 (`--headed`와 동일 효과)

## 데이터 흐름

1. `scrape_kbo_attendance(years, headless=…)`로 연도별 GraphDaily 행 수집  
2. `scraped_rows_to_dataframe` → `enrich_attendance_df`로 컬럼 보강  
3. 구장명 정규화 후 해당 구장만 필터  
4. `before`가 있으면 그 시점 이전 경기만 남기고 `경기날짜` 내림차순으로 **최근 N경기** 선택 후 오름차순 정렬  

## 주의 사항

- **네트워크·KBO 사이트 구조 변경** 시 스크랩이 실패할 수 있습니다.  
- **구장 표기**는 학습·앱과 맞도록 `stadium_aliases`에 맞춰 주세요.  
- 결과가 비면 종료 코드 `1`과 함께 stderr에 안내 메시지가 출력됩니다.

## 관련 브랜치 문서

- 전처리·학습·EDA 예시는 `develop` 등에서 `feat/eda`, `feat/ml-modeling` 섹션을 참고하면 됩니다.

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com   
