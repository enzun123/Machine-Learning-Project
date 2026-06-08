# CSV 일괄 관중 예측 (`feat/streamlit-csv-batch-predict`)

**브랜치:** `feat/streamlit-csv-batch-predict`

Streamlit에서 **경기 일정 CSV**를 업로드하면 **피처를 자동 생성**하고 RF/LGBM/XGB로 **관중 수를 일괄 예측**합니다. 단일 경기 UI와 **동일한 추론 로직**(`batch_feature_builder`)을 쓰며, **대체 구장(포항·울산·청주)·구장 수용인원** 처리를 통합합니다.

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **UI** | `scripts/app/csv_batch_predict_ui.py` — Streamlit 일괄 예측 탭 |
| **피처 생성** | `scripts/modeling/batch_feature_builder.py` — 일정 → `kbo_train_ready`와 동일 스키마 |
| **추론** | `scripts/modeling/batch_predict.py` — joblib 배치 예측 |
| **일정 CSV** | `경기날짜, 홈팀, 방문팀, 구장` (+ 선택: 기온·강수·습도·풍속) |
| **시점** | 선택한 **경기 날짜 이전** `kbo_train_ready`만 사용 (폼·prior 누수 방지) |

---

## 변경·추가 파일

| 파일 | 내용 |
|------|------|
| `scripts/app/csv_batch_predict_ui.py` | CSV 업로드·미리보기·일괄 예측 UI (**신규 → 개선**) |
| `scripts/modeling/batch_feature_builder.py` | `build_features_from_schedule()`, `read_schedule_csv_bytes()` |
| `scripts/modeling/batch_predict.py` | `predict_batch()` CLI·UI 공용 |
| `scripts/common/stadium_capacity.py` | 구장별 정원·예측 상한 clip |
| `scripts/common/secondary_venue_stats.py` | 대체 구장 prior 보강 |
| `data/external/batch_predict_schedule_template.csv` | 일정 템플릿 |
| `data/external/batch_predict_feature_template.csv` | 피처 컬럼 참고용 |

---

## CSV 형식

### 일정 (필수)

```csv
경기날짜,홈팀,방문팀,구장
2026-03-28,SSG,KIA,문학
2026-03-28,LG,KT,잠실
```

- **템플릿:** `data/external/batch_predict_schedule_template.csv`
- 컬럼 별칭: `날짜`/`date`, `홈`/`home`, `원정`/`away` 등 (`batch_feature_builder`에서 매핑)
- **선택 컬럼:** `기온`, `강수`, `습도`, `풍속(m/s)` — 없으면 UI 기본값 사용

### 피처 (자동 생성)

업로드 일정 → `build_features_from_schedule()` → 모델 `FEATURE_COLUMNS`와 동일 행.  
참고 스키마: `data/external/batch_predict_feature_template.csv`

---

## 대체 구장·수용인원

| 구장 | 처리 |
|------|------|
| **포항·울산·청주** | `secondary_venue_stats` prior + OHE는 홈 본구장 (`stadium_for_model_ohe`) |
| **정원 clip** | `venue_clip_capacity()` — KIA 광주 등 상한 반영 |
| **평가** | `evaluate_model.py` — `구장_actual` 기준 MAE 보조 (`d00e46c`) |

---

## 실행 방법

### Streamlit (UI)

```bash
cd machine-learning-project
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

앱에서 **CSV 일괄 예측** 섹션 → 템플릿 다운로드·업로드 → ML 알고리즘 선택 → 실행.

### CLI (배치만)

```bash
cd machine-learning-project
python3 scripts/modeling/batch_predict.py --schedule path/to/schedule.csv
```

**사전 조건**

- `data/processed/kbo_train_ready.csv`
- `models/attendance_*_pipeline.joblib` (벤치마크 또는 `train_model.py`)

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `8c97120` | CSV 업로드 일괄 예측, `batch_feature_builder`·`batch_predict` 도입 |
| `69b134b` | UI 개선, 대체 구장·수용인원 통합 |
| `d00e46c` | 대체 구장 prior 보강, `구장_actual` 평가 |
| `36f4510` | 벤치마크·평가 README 정리 (**브랜치 tip 근처**) |

---

## 상위·후속 브랜치

- **`feat/merge-secondary-stadiums`** — 12→9구장 OHE 통합  
- **`feat/streamlit-csv-batch-predict`** — CSV 일괄 예측 (**현재**)  
- **`feat/ui-fix`** — CSV·단일 UI 레이아웃 마감

---

## 한계

- 학습 데이터 **2024–25** 기준 — **2026·미래** 일정은 오차 클 수 있음  
- 날씨 미입력 시 UI 기본값·버킷으로 대체 (실제 경기일 기상과 다를 수 있음)
