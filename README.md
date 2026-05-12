# KBO 관중 + 기상 API 병합 (`feat/weather-api`)

**브랜치:** `feat/weather-api`

`data/raw/`의 **관중 스크랩 CSV**를 읽어, 경기일·구장에 대응하는 **기상청 지상관측 API**에서 일별 기온·강수·풍속·습도를 붙인 뒤, **`data/interim/`** 에 `*_attendance_weather.csv` 로 저장합니다. API 호출 결과는 **`weather_cache.json`** 에 누적해 같은 날짜·지점은 재요청을 줄입니다.

## 브랜치 개요

- **입력(기본):**  
  - `machine-learning-project/data/raw/kbo_2024_attendance.csv`  
  - `machine-learning-project/data/raw/kbo_2025_attendance.csv`  
  (`feat/scraping-kbo`의 `kbo_scraping.py` 산출물과 이름을 맞추는 것을 권장)
- **출력:**  
  - `machine-learning-project/data/interim/kbo_2024_attendance_weather.csv`  
  - `machine-learning-project/data/interim/kbo_2025_attendance_weather.csv`  
- **캐시:** `machine-learning-project/data/external/weather_cache.json`  
- **추가 컬럼 예:** `지점번호`, `관측도시`, `일평균기온(°C)`, `일합계강수량(mm)`, `일평균풍속(m/s)`, `일평균상대습도(%)`  
- **구장→관측소:** 스크립트 내 `STADIUM_STN_MAP` (`kbo_scraping.py`의 `기상_매핑_지역키`와 같은 권역을 쓰도록 유지)

## 전체 코드 실행 순서

1. `feat/scraping-kbo` — `data/raw/kbo_*_attendance.csv`  
2. **`feat/weather-api`** → interim `*_attendance_weather.csv`  
3. `feat/preprocessing` — `final_dataset.csv`  
4. `feat/stadium-capacity` … 이하 동일  

## 주요 구조

```text
machine-learning-project/
├── data/
│   ├── raw/                         # 입력 관중 CSV
│   ├── interim/                     # ★ 병합 결과
│   └── external/
│       └── weather_cache.json       # (일자, 지점)별 API 응답 캐시
└── scripts/data_collection/
    └── weather_api.py               # ★ 진입점
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| API | 기상청 API허브 일별 통계 엔드포인트(기온·강수·풍속·습도) |
| 캐시 | `날짜YYYYMMDD_지점번호` 키로 JSON 병합 저장, 재실행 시 누락 분만 요청 |
| 습도 보간 | 원천 결측 시 인접 일자 캐시로 보완(`impute_humidity_from_neighbors`) |
| 결측 코드 | `-9`, `-99.9` 등 관측 결측 표기는 빈 값으로 정리(`clean_weather_value`) |
| 호출 간격 | 요청 사이 `time.sleep(0.3)` — API 부하·차단 완화 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install requests
```

### 2. 스크립트 실행

`data/raw/`에 위 기본 파일명이 있어야 합니다.

```bash
cd machine-learning-project
python3 scripts/data_collection/weather_api.py
```

저장소 루트에서:

```bash
python3 machine-learning-project/scripts/data_collection/weather_api.py
```

로그에 로드·수집 진행·`interim` 저장 파일명이 출력됩니다. **입력 CSV가 하나도 없으면** 처리 없이 종료합니다.

### 3. (선택) `PYTHONPATH`

다른 위치에서 모듈을 import 할 경우 저장소 루트에서:

```bash
export PYTHONPATH=machine-learning-project/scripts
```

일반적으로는 위 **단일 스크립트 실행**만으로 충분합니다.

## 데이터 흐름

1. raw 관중 CSV별로 행 로드  
2. 각 행의 `구장`·`경기날짜`(또는 `날짜`)로 `(YYYYMMDD, 지점번호)` 집합 생성  
3. 캐시에 없는 (날짜, 지점)×변수만 API 요청 후 캐시 갱신  
4. 원본 행에 관측소 메타·기상 컬럼을 붙여 `interim`에 `*_weather.csv` 로 기록  

## 주의 사항

- **API 인증키:** 현재 스크립트에 키가 포함되어 있습니다. 공개 저장소라면 **환경 변수 등으로 이전**하고, 키 회전·유출 시 즉시 재발급하세요.  
- **네트워크:** 기상청 API 장애·응답 형식 변경 시 파싱 실패가 날 수 있습니다.  
- **구장명:** `STADIUM_STN_MAP`에 없는 `구장` 표기는 지점이 비어 기상 컬럼이 빈 채로 남을 수 있습니다. 스크랩 단계 별칭과 맞춰 주세요.  
- **이용 약관:** 공공 API 호출 정책·일일 한도를 준수하세요.

## 관련 브랜치

- **`feat/scraping-kbo`:** raw 입력 생성  
- **`feat/preprocessing`:** interim → `final_dataset.csv`  
- **`feat/stadium-capacity`:** 구장 정원 정본(기상과 별개이나 동일 `구장` 키 계약)  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
