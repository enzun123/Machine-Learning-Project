# KBO 관중 예측 — 탐색적 데이터 분석 (`feat/eda`)

**브랜치:** `feat/eda`

`final_dataset.csv`와 구장 정원 정보를 바탕으로 **관중 수 분포·범주형 요약·정원 대비 수용률·기상 변수와의 관계**를 그림과 마크다운 요약으로 남깁니다. **모델 학습용 테이블(`kbo_train_ready.csv`)이나 피처 스키마는 이 브랜치에서 바꾸지 않습니다** — 아이디어·리포트용입니다.

## 브랜치 개요

- **입력**
  - `machine-learning-project/data/processed/final_dataset.csv`  
    필수 컬럼 예: `관중수`, `요일`, `홈팀`, `구장`, `일평균기온(°C)`, `일합계강수량(mm)`, `일평균풍속(m/s)`, `일평균상대습도(%)`
  - `machine-learning-project/data/external/kbo_stadium_info.csv` — `구장`, `최대수용인원`
- **처리:** `common.stadium_aliases.STADIUM_ALIAS`로 구장명 통일(전처리·`build_features`와 동일 정본)
- **산출:** 실행할 때마다 같은 경로에 **덮어쓰기**
  - `machine-learning-project/reports/eda/figures/*.png` (7종)
  - `machine-learning-project/reports/eda/eda_summary.md`

## 전체 코드 실행 순서

1. `feat/scraping-kbo`  
2. `feat/weather-api`  
3. `feat/preprocessing`  
4. `feat/stadium-capacity` — `kbo_stadium_info.csv`  
5. **`feat/eda`** 
6. `feat/feature-engineering` — **`kbo_train_ready.csv`·학습 컬럼 확정**  
7. `feat/ml-modeling`  
8. `feat/streamlit-ui`  

## 주요 구조

```text
machine-learning-project/
├── data/
│   ├── processed/final_dataset.csv    # 필수 입력
│   └── external/kbo_stadium_info.csv  # 정원 조인용
├── scripts/
│   ├── common/stadium_aliases.py
│   └── eda/run_eda.py                 # ★ 이 브랜치 진입점
└── reports/eda/
    ├── eda_summary.md
    └── figures/
        ├── 01_target_attendance_distribution.png
        ├── 02_categorical_barplots.png
        ├── 03_categorical_boxplots.png
        ├── 04_stadium_capacity_ratio.png
        ├── 05_weather_correlation_heatmap.png
        ├── 06_rainy_vs_non_rainy.png
        └── 07_temperature_scatter.png
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| 타깃 분포 | 관중 수 히스토그램+KDE, 기초 통계량 |
| 범주형 | 요일·홈팀·구장별 평균 막대, 박스플롯 |
| 수용률 | 구장별 평균 관중 / 최대수용인원 |
| 기상 | 관중과의 상관 히트맵, 강수 여부 박스플롯, 기온–관중 산점도+회귀선 |
| 요약 MD | 위 결과를 숫자 위주로 `eda_summary.md`에 정리, 피처 제안 문단 포함 |

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install pandas matplotlib seaborn
```

### 2. EDA 실행

**권장:** `machine-learning-project` 디렉터리에서 (스크립트 docstring과 동일)

```bash
cd machine-learning-project
python3 scripts/eda/run_eda.py
```

저장소 바깥 루트(`Machine-Learning-Project`)에만 있을 때:

```bash
python3 machine-learning-project/scripts/eda/run_eda.py
```

완료 시 터미널에 `reports/eda` 경로와 `eda_summary.md` 위치가 출력됩니다.

### 3. (선택) 모듈 경로

`run_eda.py`가 실행 시 `scripts`를 `sys.path`에 넣어 `common`을 import 합니다. 별도 REPL에서 `common`만 쓰려면 저장소 루트에서:

```bash
export PYTHONPATH=machine-learning-project/scripts
```

## 데이터 흐름

1. `final_dataset.csv`·`kbo_stadium_info.csv` 로드 및 필수 컬럼 검사  
2. 구장명을 `STADIUM_ALIAS`로 치환 후 병합·집계  
3. 단계별 시각화를 `figures/`에 PNG로 저장  
4. 집계 수치를 바탕으로 `eda_summary.md` 생성  

## 주의 사항

- **Mac 기준** 스크립트에 `AppleGothic` 폰트 설정이 들어 있습니다. 다른 OS에서는 폰트만 조정하면 됩니다.  
- **`kbo_train_ready`:** 이 스크립트는 생성·수정하지 않습니다. 학습 파이프라인은 `feat/feature-engineering` 이후를 따릅니다.  
- **Git에 리포트 올릴지**는 팀 정책에 따르면 됩니다(매 실행 시 덮어쓰기).

## 관련 브랜치

- **`feat/preprocessing` 등:** `final_dataset.csv` 생성 경로  
- **`feat/feature-engineering`:** EDA에서 나온 아이디어를 반영해 학습용 피처 테이블 확정  
- **`feat/ml-modeling`:** `kbo_train_ready.csv`로 학습  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
