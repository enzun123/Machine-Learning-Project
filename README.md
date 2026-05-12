# KBO 구장 수용 인원 정본 (`feat/stadium-capacity`)

**브랜치:** `feat/stadium-capacity`

팀이 유지하는 **구단·구장별 최대 수용 인원** 목록(`STADIUM_ROWS`)을 바탕으로 **`kbo_stadium_info.csv`** 를 생성합니다. 이 CSV는 EDA, `build_features`(정원·소규모 구장 플래그), Streamlit 등에서 **동일 정본**으로 조인됩니다. 경기 데이터의 **구장 문자열 표기**는 `common/stadium_aliases.STADIUM_ALIAS`로 `kbo_stadium_info`의 `구장` 열과 맞춥니다.

## 브랜치 개요

- **진입 스크립트:** `machine-learning-project/scripts/data_collection/kbo_size.py`  
- **산출:** `machine-learning-project/data/external/kbo_stadium_info.csv`  
  컬럼: `구단`, `구장`, `최대수용인원` (UTF-8 BOM)  
- **별칭 정본:** `machine-learning-project/scripts/common/stadium_aliases.py`의 `STADIUM_ALIAS`  
  (원천 CSV/스크랩에만 나오는 표기 → 표준 구장명)

## 전체 코드 실행 순서

1. `feat/scraping-kbo`  
2. `feat/weather-api`  
3. `feat/preprocessing`  
4. **`feat/stadium-capacity`**  (`kbo_stadium_info.csv`)  
5. `feat/eda`  
6. `feat/feature-engineering`  
7. `feat/ml-modeling`  
8. `feat/streamlit-ui`  

## 주요 구조

```text
machine-learning-project/
├── data/external/
│   └── kbo_stadium_info.csv       # ★ kbo_size.py 산출
└── scripts/
    ├── common/
    │   └── stadium_aliases.py     # 구장명 별칭 (수동 유지)
    └── data_collection/
        └── kbo_size.py            # ★ STADIUM_ROWS → CSV
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| 단일 소스 | `kbo_size.py` 상단 `STADIUM_ROWS`만 수정 후 스크립트 재실행으로 CSV 갱신 |
| 구단·구장 행 | 동일 구장이라도 구단별로 행이 나뉠 수 있음(잠실 LG/두산 등) — 조인 정책은 하류 스크립트에서 `구장` 기준으로 집계 |
| 별칭 | CSV에 없는 표기(예: 한밭→대전)는 `STADIUM_ALIAS`에서 통일 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas
```

### 2. CSV 생성·갱신

1. `kbo_size.py`의 **`STADIUM_ROWS`** 를 KBO/구장 공지 등 근거에 맞게 수정  
2. 아래 실행:

```bash
cd machine-learning-project
python3 scripts/data_collection/kbo_size.py
```

저장소 루트에서:

```bash
python3 machine-learning-project/scripts/data_collection/kbo_size.py
```

성공 시 `저장 완료:`와 행 수, 별칭 파일 안내가 stdout에 출력됩니다.

### 3. 별칭을 바꿀 때

원천 데이터에 **새 구장 표기**가 생기면 `stadium_aliases.py`의 `STADIUM_ALIAS`에 추가·수정하고, EDA·피처·UI에서 구장 키가 기대와 맞는지 확인합니다.

## 데이터 흐름

1. `STADIUM_ROWS` → `DataFrame`  
2. `data/external/kbo_stadium_info.csv`로 저장  
3. 하류에서 `구장`(+별칭 치환)으로 `최대수용인원` 맵 조인  

## 주의 사항

- **수용 인원은 시즌·리모델링으로 변동**할 수 있습니다. 수치 변경 시 근거(연도·출처)를 커밋 메시지나 팀 노트에 남기는 것을 권장합니다.  
- **CSV와 별칭 불일치:** `STADIUM_ALIAS`에만 있고 CSV `구장`에 대응 행이 없으면 하류에서 정원이 비거나 폴백 로직으로 갈 수 있으니, 추가 시 **양쪽**을 함께 점검하세요.

## 관련 브랜치

- **`feat/eda`:** `kbo_stadium_info.csv` 조인·수용률 시각화  
- **`feat/feature-engineering`:** `stadium_capacity`, `is_small_stadium` 등  
- **`feat/scraping-kbo` / `feat/preprocessing`:** 구장 문자열이 별칭·정본과 맞는지 상호 검증  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
