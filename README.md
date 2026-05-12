# KBO 관중 예측 — 피처 엔지니어링 (`feat/feature-engineering`)

**브랜치:** `feat/feature-engineering`

전처리된 **`final_dataset.csv`** 를 읽어 날씨·구장·요일·시즌 형태·승률/순위(선택)·역대 폼 등을 한 번에 가공하고, **`train_model.py`가 기대하는 컬럼 집합(`MODEL_READY_COLUMNS`)** 으로 정리된 **`kbo_train_ready.csv`** 를 생성합니다. 이 단계가 끝나야 `feat/ml-modeling`에서 학습을 돌릴 수 있습니다.

## 브랜치 개요

- **필수 입력:** `machine-learning-project/data/processed/final_dataset.csv`  
  스크립트가 검사하는 필수 컬럼: `연도`, `월`, `주차_ISO`, `홈팀`, `방문팀`, `구장`, `요일`, `경기날짜`, `일합계강수량(mm)`, `일평균기온(°C)`, `일평균풍속(m/s)`, `일평균상대습도(%)`, `관중수`
- **권장 입력:** `machine-learning-project/data/external/kbo_stadium_info.csv` (`구장`, `최대수용인원`) — 없으면 정원은 결측 처리 후 평균으로 보간
- **선택 입력:** `machine-learning-project/data/external/kbo_standings_daily.csv` — 없으면 승률·게임차 관련 피처는 기본값(예: 승률 0.5)으로 채움
- **산출:** `machine-learning-project/data/processed/kbo_train_ready.csv` (UTF-8 BOM, `MODEL_READY_COLUMNS`만 저장)

## 전체 코드 실행 순서

1. `feat/scraping-kbo`  
2. `feat/weather-api`  
3. `feat/preprocessing` — `final_dataset.csv` 생성 (`build_features` 실패 시 안내하는 `preprocess_attendance_weather.py` 등)  
4. `feat/stadium-capacity` — `kbo_stadium_info.csv`  
5. `feat/eda`  
6. **`feat/feature-engineering`**  
7. `feat/ml-modeling`  
8. `feat/streamlit-ui`  

## 주요 구조

```text
machine-learning-project/
├── data/
│   ├── processed/
│   │   ├── final_dataset.csv       # 필수 입력
│   │   └── kbo_train_ready.csv     # ★ 산출
│   └── external/
│       ├── kbo_stadium_info.csv    # 권장
│       └── kbo_standings_daily.csv # 선택 (일자·팀별 승률/게임차)
├── scripts/
│   ├── common/stadium_aliases.py
│   └── features/build_features.py  # ★ 진입점
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| 구장 정규화 | `STADIUM_ALIAS`로 구장명 통일 후 정원 맵 조인 |
| 날씨 버킷 | 강수 구간(`rain_bucket`), 기온·습도·풍속 구간, `is_rain` / `is_hot` |
| 요일·주기 | `is_weekend`, 금/토/일 플래그, `weekday_sin` / `weekday_cos` |
| 이벤트·매치업 | 개막 홈전 `is_season_opener`, 어린이날 전후 `is_childrens_day`, 더비 `is_derby` |
| 순위·승률 | `kbo_standings_daily.csv` 있으면 경기일 기준 조인, 없으면 안전한 기본값 |
| 폼·무승부 proxy | 시계열 순으로 누적(행 단위 관중 누수 없음), `matchup_prior_mean_att`, `season_progress` 등 |
| 타깃 정리 | 정원의 105% 초과 관중은 소프트 클리핑 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas numpy
```

### 2. 피처 생성

저장소 루트(`Machine-Learning-Project`)에서:

```bash
python3 machine-learning-project/scripts/features/build_features.py
```

또는:

```bash
cd machine-learning-project
python3 scripts/features/build_features.py
```

성공 시 `kbo_train_ready.csv` 경로와 적용된 개선 이력 요약이 터미널에 출력됩니다.

### 3. (선택) `PYTHONPATH`

`build_features.py`가 실행 시 `scripts`를 `sys.path`에 넣어 `common`을 불러옵니다. REPL에서만 모듈을 직접 import 할 때는 저장소 루트에서:

```bash
export PYTHONPATH=machine-learning-project/scripts
```

## 데이터 흐름

1. `final_dataset.csv` 로드 → 필수 컬럼 검사  
2. 구장 CSV가 있으면 `stadium_capacity`·`is_capacity_missing` 부여  
3. 기상·요일·이벤트·더비 플래그 생성  
4. `join_standings_features`로 순위 데이터 병합(파일 없으면 기본값)  
5. `add_season_form_and_draw_proxy`로 팀/매치업 과거 요약 피처 추가  
6. `MODEL_READY_COLUMNS`만 추려 `kbo_train_ready.csv`로 저장  

## 주의 사항

- **`final_dataset.csv`가 없으면** 스크립트가 전처리 스크립트 실행을 안내하고 종료합니다.  
- **`kbo_standings_daily.csv` 스키마**가 맞지 않으면 조인 분기에서 기본값으로 떨어질 수 있으니, 승률·게임차 피처를 쓰려면 컬럼(`기준일`, `팀명`, `승률`, `게임차`, `순위` 등)을 스크립트와 맞춥니다.  
- **`train_model.FEATURE_COLUMNS`** 와 이 브랜치의 `MODEL_READY_COLUMNS`가 어긋나면 학습 단계에서 `KeyError`가 납니다. 피처를 바꾼 뒤에는 `train_model.py`와 함께 검증하세요.

## 관련 브랜치

- **`feat/preprocessing`:** `final_dataset.csv` 상류  
- **`feat/eda`:** 스키마·분포 참고 (이 스크립트는 EDA 산출물을 직접 읽지 않음)  
- **`feat/ml-modeling`:** `kbo_train_ready.csv` 소비  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
