"""
KBO 일자별 관중 기록 스크래핑 (GraphDaily)
- Target: 관중수
- 피처: 경기 날짜·시간, 요일/평일·주말, 홈·방문, 구장, 기상 매핑용 지역키, 월·주차 등
"""
from __future__ import annotations

import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

# 구장명(부분 일치) → 기상·지역 데이터 조인용 키
STADIUM_TO_REGION_KEY: dict[str, str] = {
    "잠실": "서울_잠실",
    "고척": "서울_고척",
    "수원": "경기_수원",
    "인천": "인천",
    "문학": "인천",
    "사직": "부산",
    "대전": "대전",
    "광주": "광주",
    "대구": "대구",
    "창원": "경남_창원",
    "울산": "울산",
}


def stadium_to_region_key(stadium: str) -> str:
    s = str(stadium).strip()
    for needle, key in STADIUM_TO_REGION_KEY.items():
        if needle in s:
            return key
    return "기타"


def weekday_to_bucket(day_label: str) -> str:
    """요일 한 글자(또는 괄호 안 표기) 기준 평일/주말."""
    t = re.sub(r"\s+", "", str(day_label))
    m = re.search(r"([월화수목금토일])", t)
    if not m:
        return "미상"
    d = m.group(1)
    return "주말" if d in ("토", "일") else "평일"


def _strip_parenthesized_weekday(date_cell: str) -> str:
    """날짜 셀에 '(화)' 등이 붙어 있는 경우 제거."""
    return re.sub(r"\s*[\(（][^)）]*[\)）]", "", str(date_cell).strip()).strip()


def parse_game_datetime(date_str: str, time_str: str, season_year: int) -> pd.Timestamp:
    """
    KBO 표 날짜·시간 파싱.
    날짜: 'YYYY/MM/DD'(일별 관중), 'YYYY.MM.DD', 'YY/MM/DD', 'MM.DD' 등
    시간: '18:30' 형태만 반영. 팀명 등 비시간 문자열은 무시.
    season_year: 시즌 선택(드롭다운) 연도 — 'MM.DD'만 올 때 연도 결정에 사용.
    """
    d = _strip_parenthesized_weekday(date_str)
    parts = [p.strip() for p in re.split(r"[./-]", d) if p.strip()]

    y, month, day = None, None, None
    if len(parts) == 3:
        p0 = parts[0]
        month, day = int(parts[1]), int(parts[2])
        if p0.isdigit() and len(p0) == 2:
            y = int(p0) + 2000
            if abs(y - season_year) > 1:
                y = season_year
        else:
            y = int(p0)
            if y < 100:
                y += 2000
    elif len(parts) == 2:
        month, day = int(parts[0]), int(parts[1])
        y = season_year
    else:
        return pd.NaT

    tt = str(time_str).strip()
    hm = re.search(r"(\d{1,2})\s*:\s*(\d{2})", tt)
    hour, minute = 0, 0
    if hm:
        hour, minute = int(hm.group(1)), int(hm.group(2))

    try:
        return pd.Timestamp(year=y, month=month, day=day, hour=hour, minute=minute)
    except (ValueError, TypeError):
        return pd.NaT


def split_away_home(game_cell: str) -> tuple[str, str]:
    raw = str(game_cell).strip()
    for sep in (" vs ", " VS ", "vs", "VS"):
        if sep in raw:
            a, b = raw.split(sep, 1)
            return a.strip(), b.strip()
    return "", ""


def _read_table_headers(table) -> list[str]:
    """thead th 또는 첫 헤더 행에서 열 이름 수집."""
    thead = table.find("thead")
    if thead:
        tr = thead.find("tr")
        if tr:
            texts = [th.get_text(strip=True) for th in tr.find_all("th")]
            if texts:
                return texts
    for tr in table.find_all("tr", limit=5):
        ths = tr.find_all("th")
        if ths:
            return [th.get_text(strip=True) for th in ths]
    return []


def _zip_header_row(headers: list[str], cells: list[str]) -> dict[str, str]:
    """헤더 길이에 맞춰 행 정렬(선행 순번 열 1칸 제거 등)."""
    h = [str(x).strip() for x in headers if str(x).strip()]
    c = list(cells)
    if h and len(c) == len(h) + 1:
        c = c[1:]
    while len(c) < len(h):
        c.append("")
    return {name: val.strip() for name, val in zip(h, c[: len(h)])}


def detect_row_schema(sample: list[str], headers: list[str]) -> str:
    """
    GraphDaily 최신 표: 날짜, 요일, 홈, 방문, 구장, 관중수 (시간·vs 텍스트 없음)
    구버전: 날짜, 요일, 시간, 원정vs홈, 구장, 관중수
    """
    hn = [re.sub(r"\s+", "", str(x)) for x in headers]
    if headers and any("홈" in x for x in hn) and any("방문" in x for x in hn):
        return "dict_header"
    if headers:
        return "dict_header"
    if sample and len(sample) >= 6:
        if re.search(r"\d{1,2}\s*:\s*\d{2}", sample[2]):
            return "legacy_list"
        if re.search(r"(?i)vs", sample[3]):
            return "legacy_list"
        return "home_away_dict"
    return "legacy_list"


def click_search_button(driver: webdriver.Chrome) -> None:
    """
    연도 드롭다운 변경 후 검색/조회 버튼을 눌러 결과를 갱신한다.
    버튼 텍스트/값이 다를 수 있어 후보를 순차 탐색한다.
    """
    # 먼저 명시적 id/class 후보를 시도
    for selector in (
        "a[id*='btnSearch']",
        "input[id*='btnSearch']",
        "button[id*='btnSearch']",
        "a[class*='btnSearch']",
        "button[class*='btnSearch']",
    ):
        elems = driver.find_elements(By.CSS_SELECTOR, selector)
        if elems:
            driver.execute_script("arguments[0].click();", elems[0])
            time.sleep(2)
            return

    # 일반 버튼/링크에서 텍스트로 검색
    candidates = driver.find_elements(By.CSS_SELECTOR, "a, button, input[type='button'], input[type='submit']")
    for elem in candidates:
        label = (elem.text or elem.get_attribute("value") or "").strip()
        if any(k in label for k in ("검색", "조회", "보기")):
            driver.execute_script("arguments[0].click();", elem)
            time.sleep(2)
            return

    # 버튼이 안 잡히면 드롭다운 change 이벤트로 fallback
    season_select = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlSeason']")
    driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}));", season_select)
    time.sleep(2)


def enrich_attendance_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """스크래핑 원본 → 모델용 컬럼 정리."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out.insert(0, "연도", year)

    if "홈" in out.columns and "방문" in out.columns:
        out["홈팀"] = out["홈"].astype(str).str.strip()
        out["방문팀"] = out["방문"].astype(str).str.strip()
    elif "경기(원정vs홈)" in out.columns:
        away, home = zip(*out["경기(원정vs홈)"].map(split_away_home))
        out["방문팀"] = away
        out["홈팀"] = home
    else:
        game_key = next((k for k in out.columns if "경기" in k and "vs" in k.lower()), None)
        if game_key:
            away, home = zip(*out[game_key].map(split_away_home))
            out["방문팀"] = away
            out["홈팀"] = home
        else:
            out["홈팀"] = ""
            out["방문팀"] = ""

    if "시간" in out.columns:
        out["경기시간"] = out["시간"].astype(str).str.strip()
    else:
        out["경기시간"] = ""

    out["기상_매핑_지역키"] = out["구장"].map(stadium_to_region_key)
    out["평일_주말"] = out["요일"].map(weekday_to_bucket)

    out["경기일시"] = [
        parse_game_datetime(d, t, year) for d, t in zip(out["날짜"], out["경기시간"])
    ]
    ts = pd.to_datetime(out["경기일시"], errors="coerce")
    out["경기날짜"] = ts.dt.strftime("%Y-%m-%d")
    out["월"] = ts.dt.month
    out["주차_ISO"] = ts.dt.isocalendar().week.astype("Int64")

    out["관중수"] = (
        out["관중수"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"": pd.NA, "-": pd.NA})
    )
    out = out.dropna(subset=["관중수"])
    out["관중수"] = out["관중수"].astype(int)

    # 최종보낼 컬럼 순서 (y는 관중수)
    cols = [
        "연도",
        "경기날짜",
        "경기시간",
        "경기일시",
        "요일",
        "평일_주말",
        "월",
        "주차_ISO",
        "홈팀",
        "방문팀",
        "구장",
        "기상_매핑_지역키",
        "관중수",
    ]
    return out[cols]


def scrape_kbo_attendance(years: list[int]) -> dict[int, list]:
    """연도별 행 목록. 각 행은 dict(헤더 매칭) 또는 list(구형 6열)."""
    chrome_options = Options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = "https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx"
    driver.get(url)
    time.sleep(3)

    all_data: dict[int, list] = {y: [] for y in years}
    year_layout: dict[int, tuple[str, list[str]]] = {}

    try:
        for year in years:
            print(f"--- {year}년 데이터 수집 시작 ---")

            try:
                year_dropdown = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlSeason']")
                year_select = Select(year_dropdown)
                year_select.select_by_value(str(year))
                click_search_button(driver)
            except Exception as e:
                print(f"연도 선택 중 오류 (사이트 구조 확인 필요): {e}")
                continue

            page_num = 1
            while True:
                print(f"{year}년 - {page_num}페이지 수집 중...")

                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                table = soup.find("table", {"class": "tData"})
                if not table:
                    table = soup.find("table", {"class": "tbl"})
                if not table:
                    print("데이터 표(Table)를 찾을 수 없습니다. 수집 종료.")
                    break

                tbody = table.find("tbody")
                if not tbody:
                    print("tbody 없음. 수집 종료.")
                    break

                rows = tbody.find_all("tr")

                if year not in year_layout:
                    hdr = _read_table_headers(table)
                    sample: list[str] | None = None
                    for row in rows:
                        tds = row.find_all("td")
                        if len(tds) < 6 or "없습니다" in tds[0].text:
                            continue
                        sample = [td.text.strip() for td in tds]
                        break
                    year_layout[year] = (detect_row_schema(sample or [], hdr), hdr)

                mode, hdr = year_layout[year]

                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) < 6 or "없습니다" in cols[0].text:
                        continue

                    row_data = [col.text.strip() for col in cols]
                    crowd_text = row_data[-1]
                    if crowd_text in ("-", "0", ""):
                        continue

                    if mode == "dict_header" and hdr:
                        all_data[year].append(_zip_header_row(hdr, row_data))
                    elif mode == "home_away_dict":
                        all_data[year].append(
                            {
                                "날짜": row_data[0],
                                "요일": row_data[1],
                                "홈": row_data[2],
                                "방문": row_data[3],
                                "구장": row_data[4],
                                "관중수": row_data[5],
                            }
                        )
                    else:
                        all_data[year].append(row_data)

                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, "a.next")
                    cls = (next_button.get_attribute("class") or "") + " "
                    href = next_button.get_attribute("href") or ""
                    if "disabled" in cls or href in ("", None, "javascript:void(0);"):
                        print(f"{year}년 마지막 페이지 도달.")
                        break

                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2)
                    page_num += 1
                except Exception:
                    print(f"{year}년 모든 페이지 수집 완료.")
                    break

    except Exception as e:
        print(f"스크래핑 중 치명적인 오류 발생: {e}")
    finally:
        driver.quit()

    return all_data


def scraped_rows_to_dataframe(rows: list) -> pd.DataFrame:
    """스크래퍼가 반환한 dict 행 또는 list 행을 DataFrame으로 통일."""
    if not rows:
        return pd.DataFrame()
    if isinstance(rows[0], dict):
        return pd.DataFrame(rows)
    return raw_rows_to_dataframe(rows)


def raw_rows_to_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    """표 컬럼 수에 따라 유연하게 매핑 (6 또는 7열 등)."""
    if not rows:
        return pd.DataFrame()

    n = len(rows[0])
    base = ["날짜", "요일", "시간", "경기(원정vs홈)", "구장", "관중수"]
    if n == len(base):
        names = base
    elif n > len(base):
        names = base + [f"extra_{i}" for i in range(n - len(base))]
    else:
        names = [f"col_{i}" for i in range(n)]

    return pd.DataFrame(rows, columns=names)


if __name__ == "__main__":
    target_years = [2024, 2025]

    scraped_data_dict = scrape_kbo_attendance(target_years)

    for year, data in scraped_data_dict.items():
        if not data:
            print(f"\n{year}년 수집된 데이터가 없습니다.")
            continue

        raw_df = scraped_rows_to_dataframe(data)
        df = enrich_attendance_df(raw_df, year)

        print(f"\n--- {year}년 최종 데이터 ---")
        print(df.head())
        print(f"총 {len(df)}건의 데이터 수집 완료!")

        file_name = f"kbo_{year}_attendance_real.csv"
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        print(f"{file_name} 파일이 성공적으로 저장되었습니다.")
