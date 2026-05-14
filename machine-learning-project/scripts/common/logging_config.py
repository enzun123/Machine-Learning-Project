"""CLI·배치 스크립트용 로깅 초기화 (루트에 핸들러가 없을 때만 설정)."""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str | None = None,
    *,
    force: bool = False,
) -> None:
    """
    루트 로거에 기본 핸들러·포맷을 붙인다.
    이미 핸들러가 있으면(예: Streamlit) ``force=False``일 때는 아무 것도 하지 않는다.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return
    if level is None:
        raw = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
        level = getattr(logging, raw, logging.INFO)
    if isinstance(level, str):
        level = getattr(logging, str(level).strip().upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FMT,
        force=force,
    )
