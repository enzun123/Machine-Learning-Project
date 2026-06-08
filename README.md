# 대체 구장 통합 — 12→9구장 (`feat/merge-secondary-stadiums`)

**브랜치:** `feat/merge-secondary-stadiums`

KBO **2군·대체 구장**(울산·청주·포항) 경기를 **홈팀 본구장** 하나로 묶어 One-Hot `구장` 차원을 줄이고, 피처·모델을 **재학습**하는 브랜치입니다.  
(학습용 `구장` 범주 **12개 → 9개**)

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **대체 구장** | `울산`, `청주`, `포항` (`SECONDARY_STADIUM_NAMES`) |
| **통합 규칙** | 해당 경기 **홈팀 본구장**으로 OHE 매핑 (`HOME_STADIUM_BY_TEAM`) |
| **정원** | 실제 경기 구장 기준 유지 — OHE만 본구장으로 합침 |
| **목적** | 희소 구장 범주·샘플 부족 완화, RF 일반화 개선 |

---

## 변경 파일

| 파일 | 내용 |
|------|------|
| `scripts/common/stadium_aliases.py` | `SECONDARY_STADIUM_NAMES`, `stadium_for_model_ohe()` 추가 |
| `scripts/features/build_features.py` | `_merge_secondary_stadium_for_ohe()` — `kbo_train_ready` 생성 시 적용 |
| `scripts/app/streamlit_app.py` | 추론 시 동일 OHE 규칙 반영 |
| `data/processed/kbo_train_ready.csv` | 통합 후 피처 테이블 갱신 |
| `models/attendance_rf_pipeline.joblib` | RF 재학습 |
| `models/train_report.json`, `eval_report.json` | 학습·평가 리포트 |

---

## 매핑 예시

| 실제 구장 | 홈팀 | 모델 OHE `구장` |
|-----------|------|-----------------|
| 울산 | 롯데 | 사직 |
| 청주 | 한화 | 대전 |
| 포항 | 삼성 | 대구 |

(`stadium_for_model_ohe(구장, 홈팀)` — `scripts/common/stadium_aliases.py`)

---

## 성능 (RF, `46eb3c4` 기준)

| 지표 | 값 |
|------|-----|
| 테스트 MAE | **1,903** |
| R² | **0.748** |
| 테스트 경기 수 | 287 (시간순 hold-out) |

---

## 실행 방법

피처·모델을 처음부터 맞출 때:

```bash
cd machine-learning-project
pip install -e .

# 전처리·피처 (final_dataset → kbo_train_ready, 대체 구장 통합 포함)
python3 scripts/features/build_features.py

# RF 학습·평가
python3 scripts/modeling/train_model.py
python3 scripts/modeling/evaluate_model.py
```

Streamlit 추론 시에도 `stadium_for_model_ohe`가 적용되므로 **별도 설정 없이** 대체 구장 입력 가능.

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `aeec609` | 울산·청주·포항 → 홈 본구장 통합, `build_features`·joblib 재학습 |
| `46eb3c4` | RF 테스트 MAE 1903 / R² 0.748 (12→9구장) (**브랜치 tip**) |

---

## 상위·후속 브랜치

- **`feat/stadium-capacity`** — 구장 정원 CSV·상한  
- **`feat/merge-secondary-stadiums`** — 대체 구장 OHE 통합 (**현재**)  
- **`feat/streamlit-csv-batch-predict`** — UI에서 포항·울산·청주 prior·수용인원 처리 확장
