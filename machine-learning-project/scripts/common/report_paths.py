"""학습·벤치마크 JSON 리포트용 경로 (저장소 루트 기준 상대 경로)."""

from __future__ import annotations

from pathlib import Path


def report_relative_path(root: Path, path: Path | str) -> str:
    """다른 PC에서도 읽을 수 있도록 project root 기준 상대 경로 문자열."""
    p = Path(path)
    root_res = root.resolve()
    try:
        return str(p.resolve().relative_to(root_res))
    except ValueError:
        return str(p)
