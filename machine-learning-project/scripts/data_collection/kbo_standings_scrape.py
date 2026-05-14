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


def _parse_standings_table(soup: BeautifulSoup) -> tuple[list[list[str]], list[str]]:
    tbl = soup.find("table", class_="tData")
    if not tbl:
        return [], []
    thead = tbl.find("thead")
    headers: list[str] = []
    if thead:
        tr = thead.find("tr")
        if tr:
            headers = [th.get_text(strip=True) for th in tr.find_all("th")]
    tbody = tbl.find("tbody")
    if not tbody:
        return [], headers
    rows: list[list[str]] = []
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        if "없습니다" in tds[0] or "데이터가" in tds[0]:
            continue
        if len(tds) >= 12:
            rows.append(tds)
    return rows, headers


def fetch_standings_for_date(
    session: requests.Session,
    *,
    season_year: int,
    target: date,
) -> tuple[list[list[str]], str, list[str]]:
    """
    Returns (table_rows, headline_snippet, thead_header_labels).
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

    tbl_rows, thead_headers = _parse_standings_table(soup2)
    return tbl_rows, headline, thead_headers


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


def _zip_standings_header_row(headers: list[str], cells: list[str]) -> dict[str, str]:
    """thead 열 이름에 맞춰 td 셀을 dict로 정렬(선행 순번 열 1칸 생략 등)."""
    h = [str(x).strip() for x in headers if str(x).strip()]
    c = list(cells)
    if h and len(c) == len(h) + 1:
        c = c[1:]
    while len(c) < len(h):
        c.append("")
    return {name: val.strip() for name, val in zip(h, c[: len(h)])}


def _cell_from_header_or_index(
    mapped: dict[str, str] | None,
    tds: list[str],
    header_keys: tuple[str, ...],
    index: int,
) -> str:
    if mapped:
        for k in header_keys:
            v = mapped.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    if index < len(tds):
        return str(tds[index]).strip()
    return ""


def rows_to_records(
    기준일: str,
    시즌연도: int,
    rows: list[list[str]],
    headline: str,
    table_headers: list[str] | None = None,
) -> list[dict]:
    스냅샷일 = _snapshot_date_from_headline(headline)
    recs: list[dict] = []
    thead_ok = bool(table_headers and any(str(x).strip() for x in (table_headers or [])))
    for tds in rows:
        m = _zip_standings_header_row(table_headers, tds) if thead_ok else None

        s_rank = _cell_from_header_or_index(m, tds, ("순위",), 0)
        s_team = _cell_from_header_or_index(m, tds, ("팀명", "팀"), 1)
        s_games = _cell_from_header_or_index(m, tds, ("경기",), 2)
        s_w = _cell_from_header_or_index(m, tds, ("승",), 3)
        s_l = _cell_from_header_or_index(m, tds, ("패",), 4)
        s_d = _cell_from_header_or_index(m, tds, ("무",), 5)
        s_pct = _cell_from_header_or_index(m, tds, ("승률",), 6)
        s_gb = _cell_from_header_or_index(m, tds, ("게임차",), 7)
        s_l10 = _cell_from_header_or_index(m, tds, ("최근10경기", "최근10"), 8)
        s_strk = _cell_from_header_or_index(m, tds, ("연속",), 9)
        s_home = _cell_from_header_or_index(m, tds, ("홈", "홈성적"), 10)
        s_road = _cell_from_header_or_index(m, tds, ("방문", "방문성적", "원정", "원정성적"), 11)

        recs.append(
            {
                "기준일": 기준일,
                "기록실_스냅샷일": 스냅샷일 or "",
                "시즌연도": 시즌연도,
                "시리즈": "정규시즌",
                "순위": int(s_rank) if s_rank.isdigit() else s_rank,
                "팀명": s_team,
                "경기": int(s_games) if s_games.isdigit() else pd.NA,
                "승": int(s_w) if s_w.isdigit() else pd.NA,
                "패": int(s_l) if s_l.isdigit() else pd.NA,
                "무": int(s_d) if s_d.isdigit() else pd.NA,
                "승률": float(s_pct) if _float_ok(s_pct) else pd.NA,
                "게임차": float(s_gb) if _float_ok(s_gb) else pd.NA,
                "최근10경기": s_l10,
                "연속": s_strk,
                "홈성적": s_home,
                "방문성적": s_road,
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
        rows: list[list[str]] = []
        hl = ""
        thead_h: list[str] = []
        try:
            rows, hl, thead_h = fetch_standings_for_date(session, season_year=year, target=d)
        except Exception as e:
            print(
                f"[{i}/{len(dates)}] WARN {d}: 요청 실패, Session 재생성 후 1회 재시도: {e}",
                flush=True,
            )
            session = requests.Session()
            try:
                rows, hl, thead_h = fetch_standings_for_date(session, season_year=year, target=d)
            except Exception as e2:
                print(f"[{i}/{len(dates)}] ERROR {d}: {e2}", file=sys.stderr, flush=True)
                time.sleep(args.sleep)
                continue

        if not rows:
            print(f"[{i}/{len(dates)}] {d}: 표 비어 있음 — 스킵", flush=True)
            time.sleep(args.sleep)
            continue

        recs = rows_to_records(d.isoformat(), year, rows, hl, thead_h)
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
