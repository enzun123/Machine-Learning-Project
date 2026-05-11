"""
KBO 기록실「일자별 팀 순위」일 스냅샷 수집 (feat/scraping-kbo)

- 출처: https://www.koreabaseball.com/Record/TeamRank/TeamRankDaily.aspx
- 시리즈: 정규시즌(ddlSeries=0)만 조회합니다.
- 출력: data/external/kbo_standings_daily.csv (UTF-8 BOM)

조인 예시: 기준일·팀명 ↔ final_dataset 의 경기날짜·홈팀/방문팀
  (같은 날 여러 경기가 있어도 그날 순위표는 동일 스냅샷입니다.)
- 기록실_스냅샷일: 페이지 문구에서 파싱한 날짜. 경기 없는 날 등에 요청일(기준일)과 다를 수 있음.

실행 예:
  cd machine-learning-project
  python3 scripts/data_collection/kbo_standings_scrape.py --season-year 2024 \\
    --date-from 2024-03-23 --date-to 2024-10-01

  python3 scripts/data_collection/kbo_standings_scrape.py --season-year 2024 \\
    --dates-from-raw data/raw/kbo_2024_attendance.csv data/raw/kbo_2025_attendance.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "kbo_standings_daily.csv"
RANK_URL = "https://www.koreabaseball.com/Record/TeamRank/TeamRankDaily.aspx"

CTL_PREFIX = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _daterange(d0: date, d1: date) -> list[date]:
    out: list[date] = []
    cur = d0
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _dates_from_raw_csv(path: Path, season_year: int) -> list[date]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "경기날짜" not in df.columns:
        raise KeyError(f"{path} 에 '경기날짜' 컬럼이 없습니다.")
    if "연도" in df.columns:
        df = df.loc[pd.to_numeric(df["연도"], errors="coerce") == season_year]
    ts = pd.to_datetime(df["경기날짜"], errors="coerce")
    dates = sorted(ts.dropna().dt.date.unique().tolist())
    return dates


def _parse_standings_table(soup: BeautifulSoup) -> list[list[str]]:
    tbl = soup.find("table", class_="tData")
    if not tbl:
        return []
    tbody = tbl.find("tbody")
    if not tbody:
        return []
    rows: list[list[str]] = []
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        if "없습니다" in tds[0] or "데이터가" in tds[0]:
            continue
        if len(tds) >= 12:
            rows.append(tds)
    return rows


def fetch_standings_for_date(
    session: requests.Session,
    *,
    season_year: int,
    target: date,
) -> tuple[list[list[str]], str]:
    """
    Returns (table_rows, headline_snippet).
    Each row: 순위, 팀명, 경기, 승, 패, 무, 승률, 게임차, 최근10, 연속, 홈, 방문
    """
    r = session.get(RANK_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    vs_el = soup.find("input", {"id": "__VIEWSTATE"})
    ev_el = soup.find("input", {"id": "__EVENTVALIDATION"})
    vsg_el = soup.find("input", {"id": "__VIEWSTATEGENERATOR"})
    if not vs_el or not ev_el or not vsg_el:
        raise RuntimeError("__VIEWSTATE 등 폼 필드를 찾을 수 없습니다. 사이트 구조 변경 가능성.")

    compact = target.strftime("%Y%m%d")
    data = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        "__VIEWSTATE": vs_el["value"],
        "__VIEWSTATEGENERATOR": vsg_el["value"],
        "__EVENTVALIDATION": ev_el["value"],
        CTL_PREFIX + "hfSearchYear": str(season_year),
        CTL_PREFIX + "hfSearchDate": compact,
        CTL_PREFIX + "hfPrevDate": "",
        CTL_PREFIX + "hfNextDate": "",
        CTL_PREFIX + "hfSearchSeries": "0",
        CTL_PREFIX + "ddlSeries": "0",
        CTL_PREFIX + "txtCanlendar": compact,
        CTL_PREFIX + "btnCalendarSelect": "",
    }
    r2 = session.post(RANK_URL, data=data, headers=HEADERS, timeout=30)
    r2.raise_for_status()
    soup2 = BeautifulSoup(r2.text, "html.parser")

    headline = ""
    for s in soup2.stripped_strings:
        if "기준" in s and re.search(r"\d{4}", s):
            headline = s.strip()
            break

    return _parse_standings_table(soup2), headline


def _snapshot_date_from_headline(headline: str) -> str | None:
    """'(2024년 05월08일 기준)' 형태에서 YYYY-MM-DD 추출."""
    if not headline:
        return None
    m = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", headline)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def rows_to_records(
    기준일: str,
    시즌연도: int,
    rows: list[list[str]],
    headline: str,
) -> list[dict]:
    스냅샷일 = _snapshot_date_from_headline(headline)
    recs: list[dict] = []
    for tds in rows:
        recs.append(
            {
                "기준일": 기준일,
                "기록실_스냅샷일": 스냅샷일 or "",
                "시즌연도": 시즌연도,
                "시리즈": "정규시즌",
                "순위": int(tds[0]) if tds[0].isdigit() else tds[0],
                "팀명": tds[1],
                "경기": int(tds[2]) if tds[2].isdigit() else pd.NA,
                "승": int(tds[3]) if tds[3].isdigit() else pd.NA,
                "패": int(tds[4]) if tds[4].isdigit() else pd.NA,
                "무": int(tds[5]) if tds[5].isdigit() else pd.NA,
                "승률": float(tds[6]) if _float_ok(tds[6]) else pd.NA,
                "게임차": float(tds[7]) if _float_ok(tds[7]) else pd.NA,
                "최근10경기": tds[8],
                "연속": tds[9],
                "홈성적": tds[10],
                "방문성적": tds[11],
            }
        )
    return recs


def _float_ok(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KBO 일자별 팀 순위(정규시즌) 수집")
    p.add_argument("--season-year", type=int, required=True, help="시즌 연도(예: 2024)")
    p.add_argument("--date-from", type=str, default=None, help="시작일 YYYY-MM-DD")
    p.add_argument("--date-to", type=str, default=None, help="종료일 YYYY-MM-DD (포함)")
    p.add_argument(
        "--dates-from-raw",
        type=Path,
        nargs="+",
        default=None,
        help="raw 관중 CSV 하나 이상 — 경기날짜(각 파일에서 --season-year 연도만) 합집합 후 수집",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"출력 CSV (기본: {DEFAULT_OUTPUT})",
    )
    p.add_argument("--sleep", type=float, default=0.8, help="요청 간 대기(초)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="출력 CSV에 이미 있는 (기준일, 시즌연도) 날짜는 건너뜀",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    year = args.season_year

    if args.dates_from_raw:
        date_set: set[date] = set()
        for raw in args.dates_from_raw:
            path = Path(raw)
            if not path.is_file():
                path = PROJECT_ROOT / raw
            if not path.is_file():
                print(f"FATAL: 파일 없음: {raw}", file=sys.stderr)
                sys.exit(1)
            date_set.update(_dates_from_raw_csv(path, year))
        dates = sorted(date_set)
    else:
        if not args.date_from or not args.date_to:
            print("FATAL: --date-from/--date-to 또는 --dates-from-raw 가 필요합니다.", file=sys.stderr)
            sys.exit(1)
        d0 = datetime.strptime(args.date_from, "%Y-%m-%d").date()
        d1 = datetime.strptime(args.date_to, "%Y-%m-%d").date()
        dates = _daterange(d0, d1)

    if not dates:
        print("FATAL: 수집할 날짜가 없습니다.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[tuple[str, int]] = set()
    if args.skip_existing and args.output.exists():
        old = pd.read_csv(args.output, encoding="utf-8-sig")
        if "기준일" in old.columns and "시즌연도" in old.columns:
            for _, r in old.iterrows():
                existing.add((str(r["기준일"]), int(r["시즌연도"])))

    session = requests.Session()
    all_recs: list[dict] = []

    for i, d in enumerate(dates, 1):
        key = (d.isoformat(), year)
        if key in existing:
            print(f"[{i}/{len(dates)}] skip {d} (already in output)", flush=True)
            continue
        try:
            rows, hl = fetch_standings_for_date(session, season_year=year, target=d)
        except Exception as e:
            print(f"[{i}/{len(dates)}] ERROR {d}: {e}", file=sys.stderr, flush=True)
            time.sleep(args.sleep)
            continue

        if not rows:
            print(f"[{i}/{len(dates)}] {d}: 표 비어 있음 — 스킵", flush=True)
            time.sleep(args.sleep)
            continue

        recs = rows_to_records(d.isoformat(), year, rows, hl)
        for r in recs:
            r["페이지기준문구"] = hl
        all_recs.extend(recs)
        print(f"[{i}/{len(dates)}] {d}: {len(rows)}팀 — {hl[:50] if hl else ''}", flush=True)

        time.sleep(args.sleep)

    if not all_recs:
        print("저장할 신규 행이 없습니다.", flush=True)
        return

    new_df = pd.DataFrame(all_recs)
    cols = [
        "기준일",
        "기록실_스냅샷일",
        "시즌연도",
        "시리즈",
        "순위",
        "팀명",
        "경기",
        "승",
        "패",
        "무",
        "승률",
        "게임차",
        "최근10경기",
        "연속",
        "홈성적",
        "방문성적",
        "페이지기준문구",
    ]
    new_df = new_df[cols]

    if args.output.exists() and args.skip_existing:
        prev = pd.read_csv(args.output, encoding="utf-8-sig")
        out = pd.concat([prev, new_df], ignore_index=True)
    elif args.output.exists():
        prev = pd.read_csv(args.output, encoding="utf-8-sig")
        out = pd.concat([prev, new_df], ignore_index=True)
        out = out.drop_duplicates(subset=["기준일", "시즌연도", "팀명"], keep="last")
    else:
        out = new_df

    for c in cols:
        if c not in out.columns:
            out[c] = "" if c == "기록실_스냅샷일" else pd.NA
    out = out[cols]

    out = out.sort_values(["시즌연도", "기준일", "순위"]).reset_index(drop=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("-" * 50, flush=True)
    print(f"saved: {args.output}  rows={len(out)}", flush=True)
    print("-" * 50, flush=True)


if __name__ == "__main__":
    main()
