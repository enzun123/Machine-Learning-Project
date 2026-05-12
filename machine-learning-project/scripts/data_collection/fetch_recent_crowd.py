"""
구장별 최근 N경기 관중만 수집 (KBO 기록실 GraphDaily).

이 브랜치(feature/kbo-recent-games-scrape)에서는 본 파일만 두고,
`kbo_scraping.scrape_kbo_attendance` + enrich 결과에서 구장 필터·정렬만 수행한다.

실행 예:
  cd machine-learning-project
  python3 scripts/data_collection/fetch_recent_crowd.py --stadium 창원 --n 5 -o data/cache/recent_창원.csv
  python3 scripts/data_collection/fetch_recent_crowd.py -s 대전 --before 2026-06-15 --n 5
  # → 2026-06-15 0시 이전에 치른 해당 구장 경기 중 최근 5경기
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.stadium_aliases import STADIUM_ALIAS  # noqa: E402
from data_collection.kbo_scraping import (  # noqa: E402
    enrich_attendance_df,
    scrape_kbo_attendance,
    scraped_rows_to_dataframe,
)


def _norm_stadium(name: str) -> str:
    s = str(name).strip()
    return STADIUM_ALIAS.get(s, s)


def fetch_recent_games(
    stadium: str,
    n: int = 5,
    years: list[int] | None = None,
    *,
    before: str | pd.Timestamp | None = None,
    headless: bool = True,
) -> pd.DataFrame:
    """
    지정 연도 전체 페이지를 스크랩한 뒤, 구장 기준 최근 n경기만 반환.

    before: 해당 일자(0시) **미만**인 경기만 대상으로 최근 n경기를 고른다.
            (예: 한 달 뒤 예정 경기일을 넣으면, 그날 이전에 치른 직전 5경기)
    """
    stadium = str(stadium).strip()
    if not stadium:
        return pd.DataFrame()

    y = datetime.now().year
    if years is None:
        years = [y, y - 1]

    cutoff: pd.Timestamp | None = None
    if before is not None:
        cutoff = pd.Timestamp(before).normalize()
        if pd.isna(cutoff):
            return pd.DataFrame()
        yb = int(cutoff.year)
        years = sorted(set(int(x) for x in years) | {yb, yb - 1})

    years = sorted(set(int(x) for x in years), reverse=True)

    raw = scrape_kbo_attendance(years, headless=headless)
    frames: list[pd.DataFrame] = []
    for year in sorted(set(years)):
        rows = raw.get(year) or []
        if not rows:
            continue
        raw_df = scraped_rows_to_dataframe(rows)
        if raw_df.empty:
            continue
        frames.append(enrich_attendance_df(raw_df, year))

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    need = {"경기날짜", "홈팀", "방문팀", "관중수", "구장"}
    if not need.issubset(full.columns):
        return pd.DataFrame()

    full = full.copy()
    full["_구장_norm"] = full["구장"].astype(str).map(_norm_stadium)
    needle = _norm_stadium(stadium)
    sub = full.loc[full["_구장_norm"] == needle].copy()
    if sub.empty:
        sub = full.loc[full["구장"].astype(str).str.contains(stadium, regex=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["경기날짜"] = pd.to_datetime(sub["경기날짜"], errors="coerce")
    sub = sub.dropna(subset=["경기날짜"])
    if cutoff is not None:
        sub = sub.loc[sub["경기날짜"].dt.normalize() < cutoff].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("경기날짜", ascending=False).head(int(n))
    sub = sub.sort_values("경기날짜", ascending=True)
    return sub.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="fetch_recent_crowd.py",
        description="GraphDaily → 구장별 최근 관중 N경기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python3 scripts/data_collection/fetch_recent_crowd.py -s 창원 --n 5\n"
            "  python3 scripts/data_collection/fetch_recent_crowd.py --stadium 대전 -o data/cache/recent.csv\n"
            "  python3 scripts/data_collection/fetch_recent_crowd.py -s 잠실 --years 2025 2024\n"
            "  python3 scripts/data_collection/fetch_recent_crowd.py -s 대전 --before 2026-06-15 --n 5\n"
            "\n"
            "※ --before YYYY-MM-DD : 그날 0시 '이전' 경기만 보고 그중 최근 N경기.\n"
            "※ --stadium(또는 -s)은 필수입니다. 구장명은 KBO 표기(잠실, 대전, 창원, 문학, 수원 …)에 맞춥니다."
        ),
    )
    p.add_argument(
        "-s",
        "--stadium",
        type=str,
        required=True,
        metavar="구장",
        help="구장명 (예: 창원, 대전, 잠실). 짧게 -s 창원",
    )
    p.add_argument("--n", type=int, default=5, help="경기 수 (기본 5)")
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="스크랩할 연도(미지정 시 올해·전년)",
    )
    p.add_argument("-o", "--output", type=str, default=None, help="CSV 경로 (없으면 stdout)")
    p.add_argument(
        "--before",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="기준일(해당 날 0시 미만 경기만 대상). 예: 예정 경기 한 달 뒤 날짜",
    )
    p.add_argument("--headed", action="store_true", help="브라우저 창 표시")
    args = p.parse_args()

    if args.before is not None and pd.isna(pd.Timestamp(args.before)):
        p.error(f"--before 날짜를 해석할 수 없습니다: {args.before!r}")

    if args.headed:
        os.environ["KBO_SCRAPE_HEADLESS"] = "0"

    headless = os.environ.get("KBO_SCRAPE_HEADLESS", "1") != "0"
    years = list(args.years) if args.years else None
    df = fetch_recent_games(
        args.stadium,
        n=args.n,
        years=years,
        before=args.before,
        headless=headless,
    )

    if df.empty:
        print("결과 없음. Chrome·연도·구장명을 확인하세요.", file=sys.stderr)
        sys.exit(1)

    cols = [c for c in ("연도", "경기날짜", "홈팀", "방문팀", "구장", "관중수") if c in df.columns]
    out = df[cols] if cols else df

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"저장: {args.output} ({len(out)}행)")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
