# Streamlit 최종 UI 정리 (`feat/ui-fix`)

**브랜치:** `feat/ui-fix`

Streamlit 메인·CSV 일괄 화면의 **레이아웃·알고리즘별 바 UI·피처 중요도 시각화**를 정리하는 UI 마감 브랜치입니다.

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **알고리즘 UI** | RF / LGBM / XGB 선택 결과를 **바(막대) 형식**으로 통일 |
| **CSV 일괄** | 단일·일괄 예측을 **한 화면**에 합치고 표·입력 UX 정리 |
| **피처 중요도** | permutation importance **박스플롯** 추가 |
| **마감** | `streamlit_app.py`·`csv_batch_predict_ui.py` 구조·버그 최종 수정 |

---

## 변경 파일

| 파일 | 내용 |
|------|------|
| `scripts/app/streamlit_app.py` | 단일 예측·중요도·혼잡도 레이아웃, UI 버그 수정 |
| `scripts/app/csv_batch_predict_ui.py` | 일괄 예측 UI 일관성, 한 화면 통합 |

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `32cbe62` | UI 복잡도 축소, 알고리즘별 바 형식 통일 |
| `67294c2` | CSV 예측 한 화면에 합치기 |
| `fc34b6d` | CSV 예측 UI 일관성 개선 |
| `51a9c11` | 피처 중요도 박스플롯 |
| `1d889a1` | UI 버그 수정 |
| `754a706` | 최종 UI 수정 (**브랜치 tip**) |

---

## 실행 방법

```bash
cd machine-learning-project
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

**확인 포인트**

1. 사이드바 **ML 알고리즘** 여러 개 선택 → 예측·바 차트가 모델별로 정렬  
2. **CSV 일괄 예측** 탭에서 일정 업로드 → 피처 자동 생성 → 결과 표  
3. **피처 중요도** 박스플롯 표시 (모델·데이터 있을 때)  

---

## 상위 브랜치

- **`feat/streamlit-csv-batch-predict`** — CSV 일괄 예측 기능 도입  
- **`feat/ui-fix`** — UI 마감·시각화 (**현재**)
