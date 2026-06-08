# Streamlit UI 안정화·로딩 수정 (`fix/ui2`)

**브랜치:** `fix/ui2`

Streamlit 웹앱에서 **«최근 5경기 vs 예측» 막대 차트**가 `구장` 컬럼 유무에 따라 깨지며 **무한 로딩**에 걸리던 문제를 고치고, **pandas / scikit-learn 버전**을 Cloud·로컬에서 맞춥니다.  
(`fix/light-mode-contrast`의 다크·라이트·모바일 UI 개선이 선행 merge 되어 있습니다.)

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **증상** | «최근 5경기» ON + 예측 차트 렌더 시 Streamlit 스피너가 멈추지 않거나 `ValueError: cannot set a row with mismatched columns` |
| **원인** | `chart_df`에 `구장` 컬럼이 있을 때 예측 행을 5개 값 리스트로 추가 → 열 수 불일치 |
| **해결** | `pd.concat` + dict 행으로 예측값 추가, `구장` 있으면 함께 기록 |
| **부가** | `pd.to_datetime(..., format="mixed")`, `scikit-learn==1.8.0`, Python `<3.14` |

---

## 변경 파일

| 파일 | 내용 |
|------|------|
| `scripts/app/streamlit_app.py` | `chart_df` 예측 행 `pd.concat`·dict 방식, 날짜 파싱 `format="mixed"` |
| `scripts/app/csv_batch_predict_ui.py` | 동일 날짜 파싱 정리 |
| `requirements.txt`, `pyproject.toml` | `scikit-learn==1.8.0` 고정 |
| `.python-version` | `3.12` |

---

## 선행 UI (같은 브랜치 계열)

| 커밋 | 내용 |
|------|------|
| `6906aba` | 다크·라이트 모드 — `theme.py`, `theme_watcher.py`, `styles/app.css` |
| `016db0d` | 테마 변경 후 차트 색 미반영 → **새로고침** 버튼 |
| `7d3fb55` | 모바일 시인성 CSS (`app.css`) |

---

## 실행 방법

```bash
cd machine-learning-project
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

**확인 포인트**

1. 사이드바 **«최근 5경기»** 켠 뒤 단일 경기 예측 → 막대 차트가 정상 표시  
2. Cloud 배포 시 joblib pickle 경고 없이 로드 (`sklearn 1.8.0`)  
3. 다크/라이트 전환 후 차트 색 이상 시 **새로고침** 버튼 동작  

---

## 상위 브랜치

- **`fix/light-mode-contrast`** — 다크·라이트·모바일·새로고침  
- **`fix/ui2`** — 차트 무한 로딩·버전 핀 (**현재**)

---

## 관련 이슈 (Streamlit Cloud)

동네예보 API(`_cached_forecast_rain_ref`) 호출이 길면 `Running _cached_...` 스피너가 오래 보일 수 있습니다. 본 브랜치는 **차트 DataFrame 열 불일치**로 인한 크래시·무한 대기를 우선 해결합니다.
