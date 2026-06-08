# 다크·라이트 모드·모바일 UI (`fix/light-mode-contrast`)

**브랜치:** `fix/light-mode-contrast`

Streamlit 앱에 **다크/라이트 테마**를 분리 적용하고, **모바일 시인성** CSS를 보강하며, 테마 전환 후 **matplotlib 차트 색**이 안 바뀔 때 쓸 **새로고침** 버튼을 추가합니다.

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **테마** | OS·Streamlit 설정에 맞춰 다크/라이트 CSS·차트 팔레트 자동 전환 |
| **모듈 분리** | `theme.py`, `theme_watcher.py`, `styles/app.css` |
| **새로고침** | UI 모드 변경 후 그래프 색 미반영 시 `st.rerun()` 트리거 버튼 |
| **모바일** | 좁은 화면에서 카드·바·사이드바 여백·글자 대비 조정 |

---

## 변경 파일

| 파일 | 내용 |
|------|------|
| `scripts/app/theme.py` | 라이트/다크 팔레트, `apply_figure_theme`, Streamlit inject CSS 헬퍼 |
| `scripts/app/theme_watcher.py` | 테마 변경 감지·캐시 무효화 |
| `scripts/app/styles/app.css` | 공통·다크·라이트·모바일 `@media` 규칙 |
| `scripts/app/streamlit_app.py` | 테마 연동, 새로고침 버튼, 차트 `apply_figure_theme` |
| `scripts/app/csv_batch_predict_ui.py` | 일괄 예측 UI 테마·바 스타일 통일 |

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `6906aba` | 다크·라이트 모드별 앱 UI (`theme.py` 등 신규) |
| `016db0d` | 그래프 색 미반영 → 새로고침 버튼 |
| `7d3fb55` | 모바일 UI 시인성 CSS |

---

## 실행 방법

```bash
cd machine-learning-project
pip install -e .
streamlit run scripts/app/streamlit_app.py
```

**확인 포인트**

1. macOS **시스템 설정 → 모양** 또는 Streamlit **Settings → Theme** 변경 → 앱·차트 색 동기화  
2. 모바일 폭(DevTools)에서 카드·바 레이아웃 깨짐 없음  
3. 테마 바꾼 직후 차트 색 이상 → **새로고침** 클릭 시 정상  

---

## 후속 브랜치

- **`fix/ui2`** — 최근 5경기 차트 무한 로딩·sklearn 버전 핀 (본 브랜치 위에 적용)
