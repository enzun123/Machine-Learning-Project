# KBO 2026 일정·기상 보강 및 UI (`feat/2026-schedule-and-weather`)

**브랜치:** `feat/2026-schedule-and-weather`

**2026 정규시즌**을 전제로 한 **기본 개시 시각 보강**(스크랩 표에 시간이 비었을 때), **공휴일 반영**, **기상청 API허브 동네예보(typ02)** 연동 모듈 추가, **Streamlit**에서 시간당 강수·우천 요약·동네예보 참고 표시 및 **RF permutation importance**에서 날씨 피처 고정 등을 다룹니다. 이 브랜치에서는 **`kbo_scraping.py`** 보강 로직과 함께 **원시·interim·final 데이터·모델 리포트** 일부가 갱신되어 있습니다.

## 브랜치 개요

| 구분 | 내용 |
|------|------|
| **목적** | 경기일시 누락 시 KBO 관례에 맞춘 개시 시각 채움, 혹서기(7~8월) 규칙·공휴일 반영, 예보·우천 정보를 UI·수집 파이프라인에 반영 |
| **신규·핵심 코드** | `scripts/common/kbo_regular_start_time.py`, `scripts/common/kma_vilage_fcst.py` |
| **수정** | `scripts/data_collection/kbo_scraping.py`, `scripts/data_collection/kbo_size.py`, `scripts/app/streamlit_app.py` |
| **데이터·모델** | `data/raw/*.csv`, `data/interim/*`, `data/processed/final_dataset.csv`, `data/external/kbo_stadium_info.csv`, `models/*.json` 등 (이 브랜치 커밋 기준 갱신분) |

## 전체 파이프라인에서의 위치

1. `feat/scraping-kbo` — 일별 관중 수집  
2. **`feat/2026-schedule-and-weather`** — **현재 브랜치** (스크랩 보강·동네예보 모듈·UI·관련 산출물)  
3. `feat/weather-api` — 지상 일별 기상 병합  
4. `feat/preprocessing` → `feat/feature-engineering` → `feat/ml-modeling` → `feat/streamlit-ui` …  

상류·하류 단계의 **일반 실행 순서**는 `develop` 루트 README 또는 팀 통합 문서를 참고하면 됩니다.

## 주요 구조 (이 브랜치 기준 추가·연동)

```text
machine-learning-project/scripts/
├── common/
│   ├── kbo_regular_start_time.py   # ★ 정규시즌 기본 개시 시각 + 2026 공휴일·혹서기 규칙
│   └── kma_vilage_fcst.py          # ★ 동네예보 RN1·POP 등 (API허브 typ02)
├── data_collection/
│   └── kbo_scraping.py             # 개시 시각 보강 등 스크랩 후처리 연동
└── app/
    └── streamlit_app.py            # ★ 동네예보·우천 참고, 일강수 고정 예측 흐름, RF importance 등
```

## 주요 기능

| 항목 | 설명 |
|------|------|
| `kbo_regular_start_time` | 평일 18:30, 토·일·공휴일 및 **7~8월 혹서기** 규칙(2026 시즌 연도부터 적용 등). `KR_PUBLIC_HOLIDAYS_2026` 유지보수 필요 |
| `kma_vilage_fcst` | 구장별 격자 `STADIUM_GRID`, 초단기·단기예보 조회. 인증: **`KMA_APIHUB_AUTH_KEY`** (동네예보 이용 신청) |
| 스크랩 | GraphDaily 수집 후 **시:분 공란** 시 위 모듈로 보강 |
| Streamlit | 시간당 강수·KBO 우천 요약 참고, 예측 시 **일 합계 강수 0mm 고정** 등 정책, 동네예보·휴리스틱 강수 UI, **RF permutation importance**에서 날씨 피처 고정 |
| 정원 CSV | `kbo_size.py` / `kbo_stadium_info.csv` 소폭 조정(이 브랜치 데이터 반영) |

## 환경 변수

```bash
export KMA_APIHUB_AUTH_KEY="발급받은_API허브_키"
```

- 동네예보(typ02)는 **별도 이용 신청**이 필요할 수 있습니다.  
- 키가 없으면 `kma_vilage_fcst` 일부는 `weather_api`의 기존 키 import를 시도할 수 있으나, **운영·공개 저장소에서는 환경 변수만 사용**하는 것을 권장합니다.

## 실행 방법 (이 브랜치 기능 확인용)

```bash
cd machine-learning-project

# 스크랩 (Chrome). 개시 시각 보강은 수집 파이프라인 안에서 적용
python3 scripts/data_collection/kbo_scraping.py

# UI (동네예보·우천 UI 확인 시 위 KMA 키 설정)
streamlit run scripts/app/streamlit_app.py
```

동네예보만 모듈로 시험할 때는 `PYTHONPATH=machine-learning-project/scripts` 후 `from common.kma_vilage_fcst import ...` 형태로 import 할 수 있습니다.

## 주의 사항

- **공휴일·시즌 규칙**은 행정 고시·KBO 안내 변경 시 `kbo_regular_start_time.py`를 수정해야 합니다.  
- **예보 API**는 호출 제한·응답 스펙 변경에 취약합니다.  
- 이 브랜치에 **대용량 CSV·json 갱신**이 포함되어 있으므로 PR 시 diff 크기·민감 정보 여부를 팀 규칙에 맞게 검토하세요.

## 관련 커밋 (요약)

- `feat(kbo): 2026 정규시즌 기본 개시 시각 보강 및 공휴일 반영`  
- `docs(kbo): 기본 개시 시각 모듈 docstring에 혹서기 적용 연도 명시`  
- `feat: 시간당 강수 참고·KBO 우천 요약, 예측은 일강수 0mm 고정`  
- `feat: 동네예보·우천 참고 UI, RF 중요도 날씨 고정, 휴리스틱 강수 수정`  

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
