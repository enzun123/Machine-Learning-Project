# pytest 스모크 테스트 (`feat/pytest-smoke-tests`)

**브랜치:** `feat/pytest-smoke-tests`

**네트워크·Chrome 없이** 돌릴 수 있는 **스모크 pytest 16개**를 추가합니다. `common` 모듈 동작·커밋된 데이터·RF joblib 존재 여부를 빠르게 검증합니다. macOS / Windows / CI 공통으로 사용합니다.

---

## 브랜치 개요

| 항목 | 내용 |
|------|------|
| **테스트 수** | **16개** (parametrize 포함) |
| **위치** | `machine-learning-project/tests/` |
| **의존성** | `pip install -e ".[dev]"` → `pytest>=7.0` |
| **특징** | Selenium·기상 API 호출 없음, repo에 포함된 CSV·joblib 기준 |

---

## 테스트 파일

| 파일 | 검증 내용 |
|------|-----------|
| `tests/conftest.py` | `project_root` fixture (`machine-learning-project/` 루트) |
| `tests/test_smoke_common.py` | 혼잡도, 대체 구장 OHE, KBO 개시 시각, API 키 마스킹, 우천 밴드, 구장 정원 |
| `tests/test_smoke_data.py` | `kbo_train_ready.csv`·joblib·`train_report.json` 존재, `load_training_table`, RF > dummy MAE |

### `test_smoke_common.py` 주요 케이스

- `classify_congestion_pct` — LOW / NORMAL / HIGH  
- `is_secondary_stadium`, `stadium_for_model_ohe` (울산 → 롯데 본구장)  
- `parse_row_date_only`, `default_start_hm`  
- `redact_api_secrets` — URL 내 `authKey` 마스킹  
- `rule_band_from_mm_h`, `rule_band_from_pop`  
- `venue_clip_capacity` — 대체 구장 정원 fallback  

### `test_smoke_data.py` 주요 케이스

- 필수 산출물 4종 파일 존재  
- `load_training_table` — 행 수·`관중수` 타깃  
- `train_report.json` — RF MAE < dummy MAE  

---

## 변경 파일

| 파일 | 내용 |
|------|------|
| `tests/conftest.py` | session `project_root` fixture |
| `tests/test_smoke_common.py` | common 모듈 스모크 (**신규**) |
| `tests/test_smoke_data.py` | 데이터·모델 산출물 스모크 (**신규**) |
| `pyproject.toml` | `[dev]` extra, `[tool.pytest.ini_options]` (`testpaths`, `pythonpath=scripts`) |

---

## 실행 방법

```bash
cd machine-learning-project
pip install -e ".[dev]"
pytest
```

CI와 동일하게 조용히:

```bash
pytest -q
```

**예상:** 16 passed (데이터·joblib가 repo에 있을 때). 일부 파일 없으면 해당 테스트 `skip`.

---

## pytest 설정 (`pyproject.toml`)

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["scripts"]
addopts = "-q"
```

`pip install -e .` 만으로는 pytest 미포함 → **`[dev]`** 필요.

---

## 주요 커밋

| SHA | 설명 |
|-----|------|
| `38f697e` | common·데이터 스모크 pytest 16개 추가 (**브랜치 tip**) |

---

## 상위·후속 브랜치

- **`feat/pkg-layout-and-config`** — `pyproject.toml` 패키지 레이아웃  
- **`feat/pytest-smoke-tests`** — 스모크 테스트 (**현재**)  
- **`main` / `develop`** — `.github/workflows/pytest.yml` CI (push·PR 시 Ubuntu 3.11·3.12)
