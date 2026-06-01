"""관중수 CSV·문자열 → 숫자 (엑셀 '23,000', \"23,000\" 등)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_attendance_column(s: pd.Series) -> pd.Series:
    """관중수 열 정규화."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    t = (
        s.astype(str)
        .str.strip()
        .str.strip("'\"")
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d]", "", regex=True)
    )
    t = t.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return pd.to_numeric(t, errors="coerce")


_EXTERNAL_ATTENDANCE_SKIP = frozenset(
    {
        "batch_predict_schedule_template.csv",
        "kbo_stadium_info.csv",
        "batch_predict_feature_template.csv",
    }
)


def attendance_sources_fingerprint(root: Path) -> str:
    """관중·학습 데이터 파일 변경 감지용 (세션·캐시 무효화)."""
    tokens: list[str] = []
    for pattern in (
        "data/interim/kbo_*attendance*.csv",
        "data/raw/kbo_*attendance*.csv",
        "data/kbo_*attendance*.csv",
        "data/processed/kbo_train_ready.csv",
    ):
        for path in sorted(root.glob(pattern)):
            try:
                tokens.append(f"{path.relative_to(root)}:{path.stat().st_mtime_ns}")
            except (OSError, ValueError):
                pass
    ext = root / "data" / "external"
    if ext.is_dir():
        for path in sorted(ext.iterdir()):
            if not path.is_file() or path.name in _EXTERNAL_ATTENDANCE_SKIP:
                continue
            if path.suffix.lower() not in {".tsv", ".txt", ".csv"}:
                continue
            try:
                tokens.append(f"ext/{path.name}:{path.stat().st_mtime_ns}")
            except OSError:
                pass
    return "|".join(tokens) if tokens else "empty"
