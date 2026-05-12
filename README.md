# KBO 관중 예측 — 전처리 (`feat/preprocessing`)

**브랜치:** `feat/preprocessing`

`data/interim/` 에 있는 **관중+기상** CSV를 읽어 병합·정규화한 뒤, **`final_dataset.csv`** 로 저장합니다. `feat/feature-engineering`의 `build_features.py`가 요구하는 **정렬·키 계약**(연도, 경기날짜, `game_no`, 더블헤더 순서 등)을 이 단계에서 맞춥니다.

## 브랜치 개요

- **기본 입력:**  
  - `machine-learning-project/data/interim/kbo_2024_attendance_weather.csv`  
  - `machine-learning-project/data/interim/kbo_2025_attendance_weather.csv`  
  (`feat/weather-api`의 `weather_api.py` 등으로 만들거나, **동일 스키마** CSV를 `--inputs`로 넘김)
- **출력:** `machine-learning-project/data/processed/final_dataset.csv` (기본, `--output`으로 변경 가능)
- **처리 요약:** 여러 연도 파일 `concat` → 완전 중복 제거 → 구장 `STADIUM_ALIAS` 치환 → `경기날짜`·`_ts_for_sort` 보정 → 더블헤더용 `game_no` 부여 → 관중수 파싱·결측 제거 → 정렬 후 CSV 저장

## 전체 코드 실행 순서

1. `feat/scraping-kbo`  
2. `feat/weather-api` — interim `kbo_*_attendance_weather.csv` 생성 등  
3. **`feat/preprocessing`**  → `final_dataset.csv`  
4. `feat/stadium-capacity` — `kbo_stadium_info.csv`  
5. `feat/eda`  
6. `feat/feature-engineering`  
7. `feat/ml-modeling`  
8. `feat/streamlit-ui`  

## 주요 구조

```text
machine-learning-project/
├── data/
│   ├── interim/
│   │   ├── kbo_2024_attendance_weather.csv   # 기본 입력
│   │   └── kbo_2025_attendance_weather.csv
│   └── processed/
│       └── final_dataset.csv                 # ★ 산출
└── scripts/
    ├── common/stadium_aliases.py
    ├── preprocessing/
    │   └── preprocess_attendance_weather.py  # ★ 진입점
    └── data_collection/weather_api.py        # (상류) interim 생성 예시
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| CLI | `--inputs` 여러 경로, `--output` 출력 경로 (기본 `data/processed/final_dataset.csv`) |
| 날짜 키 | `경기일시` 파싱; `경기날짜` 없으면 일시에서 `YYYY-MM-DD` 보강 (`build_features` 정렬과 동일 계약) |
| `game_no` | 동일 `(연도, 경기날짜, 홈팀, 방문팀, 구장)` 내에서 일시 순 1,2,… (더블헤더) |
| 관중수 | 쉼표·비숫자 제거 후 숫자화, 결측 행 제거, 정수 캐스팅 |
| 강수 | `일합계강수량(mm)` 결측은 0으로 채움 |
| 검증 로그 | 완전 중복 제거 건수, 관중수 이상(5만 초과·음수) 경고, 더블헤더 추정 건수 등 stdout |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas
```

### 2. 전처리 실행

interim 파일이 **기본 경로**에 있을 때 (`machine-learning-project` 기준):

```bash
cd machine-learning-project
python3 scripts/preprocessing/preprocess_attendance_weather.py
```

저장소 루트에서:

```bash
python3 machine-learning-project/scripts/preprocessing/preprocess_attendance_weather.py
```

**입력이 다른 위치일 때:**

```bash
cd machine-learning-project
python3 scripts/preprocessing/preprocess_attendance_weather.py \
  --inputs data/interim/my_2024.csv data/interim/my_2025.csv \
  --output data/processed/final_dataset.csv
```

입력이 없으면 스크립트가 stderr로 `weather_api.py` 실행·`--inputs` 지정 등 해결 방법을 안내하고 종료합니다.

### 3. (선택) `PYTHONPATH`

스크립트가 실행 시 `scripts`를 `sys.path`에 넣어 `common`을 import 합니다. REPL에서만 쓸 때는 저장소 루트에서:

```bash
export PYTHONPATH=machine-learning-project/scripts
```

## 데이터 흐름

1. 지정된 interim CSV들을 읽어 세로 병합  
2. 필수 컬럼(`경기일시`, `홈팀`, `방문팀`, `구장`, `연도` 등) 존재 확인  
3. `경기날짜`·정렬용 타임스탬프 확보 후 그룹별 `game_no` 부여  
4. 수치·강수 정리 후 `final_dataset.csv`로 UTF-8 BOM 저장  

## 주의 사항

- **interim 스키마:** 이후 `build_features`가 요구하는 컬럼(`월`, `주차_ISO`, `요일` 등)은 **상류 스크립트**가 interim에 넣어 두어야 합니다. 본 스크립트는 주로 병합·키·관중 정리에 집중합니다.  
- **입력 파일 누락:** 기본 `kbo_2024`/`2025` 경로에 파일이 없으면 즉시 실패합니다.  
- **관중수 이상치:** 5만 초과·음수는 제거하지 않고 경고만 출력합니다.

## 관련 브랜치

- **`feat/weather-api`:** interim `kbo_*_attendance_weather.csv` 생성  
- **`feat/feature-engineering`:** `final_dataset.csv` → `kbo_train_ready.csv`  
- **`feat/eda`:** `final_dataset.csv` 기반 탐색  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
