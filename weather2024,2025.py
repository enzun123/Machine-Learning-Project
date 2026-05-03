"""
KBO 경기 기상 데이터 수집기 v10 (새로운 파일명 적용 및 원본 컬럼 100% 자동 보존)
"""

import requests
import json
import csv
import time
import logging
import re
from pathlib import Path

# ─── 로깅 ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  ① 설정
# ══════════════════════════════════════════════════════════════
AUTH_KEY = "NtKXD3bRQkCSlw920cJAyA"

API_ENDPOINTS = {
    "기온": "https://apihub.kma.go.kr/api/typ01/url/sts_ta.php",
    "강수": "https://apihub.kma.go.kr/api/typ01/url/sts_rn.php",
}

WEATHER_VARS = {
    "기온": {"TA_DAVG": "일평균기온(°C)"},
    "강수": {"RN_DSUM": "일합계강수량(mm)"},
}

STADIUM_STN_MAP = {
    "잠실": 108, "두산": 108, "LG": 108,
    "고척": 108, "키움": 108, "국민": 108,
    "문학": 112, "SSG": 112,
    "수원": 119, "KT": 119,
    "대전": 133, "한밭": 133, "한화": 133,
    "포항": 138,
    "대구": 143, "삼성": 143,
    "울산": 152,
    "창원": 155, "NC": 155,
    "광주": 156, "KIA": 156,
    "사직": 159, "롯데": 159,
}

STN_CITY_MAP = {
    108: "서울",  112: "인천",  119: "수원",
    133: "대전",  138: "포항",  143: "대구",
    152: "울산",  155: "창원",  156: "광주",
    159: "부산",
}

# ★ 새롭게 올려주신 파일명으로 변경
KBO_FILES  = ["kbo_2024_attendance.csv", "kbo_2025_attendance.csv"]
CACHE_FILE = "weather_cache.json"

# ══════════════════════════════════════════════════════════════
#  ② 공통 유틸 함수
# ══════════════════════════════════════════════════════════════
def kbo_date_to_yyyymmdd(date_str: str) -> str:
    return re.sub(r"\D", "", str(date_str))

def fetch_category(category: str, date_str: str, stn_id: int) -> dict:
    params = {"tm1": date_str, "tm2": date_str, "stn_id": stn_id, "disp": 1, "help": 1, "authKey": AUTH_KEY}

    for attempt in range(3):
        try:
            resp = requests.get(API_ENDPOINTS[category], params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

            lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
            header_line, data_line = None, None

            for line in lines:
                if line.startswith("#"):
                    clean_line = line.lstrip("#").strip()
                    if clean_line and not re.match(r'^[-=]+$', clean_line) and "START" not in clean_line:
                        header_line = clean_line
                else:
                    data_line = line
                    break

            if not header_line or not data_line:
                return {}

            headers = [h for h in header_line.replace(",", " ").replace("=", " ").split() if h]
            values  = [v for v in data_line.replace(",", " ").replace("=", " ").split() if v]

            if len(headers) < 3 or len(values) < 3:
                return {}

            return dict(zip(headers, values))

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            log.warning(f"  [{category}] 연결 지연 (재시도 {attempt+1}/3) stn={stn_id} tm={date_str}...")
            time.sleep(2)
        except requests.RequestException:
            break

    return {}

def load_cache() -> dict:
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(data: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_weather(cache_entry: dict) -> dict:
    result = {}
    for cat, var_map in WEATHER_VARS.items():
        raw = cache_entry.get(cat, {})
        for api_col, out_col in var_map.items():
            val = raw.get(api_col, "")
            # 기상청 결측치 처리
            if val in ("-", "-9", "-9.0", "-9.9", "-99.9", "-999", "-999.0", "None", "nan", "="):
                result[out_col] = ""
            else:
                result[out_col] = val
    return result

# ══════════════════════════════════════════════════════════════
#  ③ 메인 실행 로직
# ══════════════════════════════════════════════════════════════
def main():
    log.info("KBO 기상 데이터 수집 시작 (원본 데이터 자동 보존 방식)")
    
    rows_by_file = {}
    unique_pairs = set()

    # 1. 파일별로 데이터 읽기 (원본 컬럼명 파악)
    for fp in KBO_FILES:
        p = Path(fp)
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                rows_by_file[fp] = {
                    "fieldnames": reader.fieldnames,
                    "rows": rows
                }
                for r in rows:
                    stn = STADIUM_STN_MAP.get(r.get("구장", "").strip())
                    # 새 파일은 '경기날짜'를 사용 (만약 예전 파일이면 '날짜' 사용)
                    date_str = r.get("경기날짜", r.get("날짜", ""))
                    if stn and date_str:
                        unique_pairs.add((kbo_date_to_yyyymmdd(date_str), stn))

    if not rows_by_file:
        log.error("처리할 CSV 파일이 없습니다. 파일명을 확인해주세요.")
        return

    unique_pairs = sorted(unique_pairs)
    cache = load_cache()
    new_pairs = [(d, s) for d, s in unique_pairs if f"{d}_{s}" not in cache]
    
    # 2. 누락된 기상 데이터 수집
    for idx, (date_str, stn_id) in enumerate(new_pairs, 1):
        key = f"{date_str}_{stn_id}"
        log.info(f"  [{idx}/{len(new_pairs)}] {date_str}  지점={stn_id}")
        cache[key] = {}
        for cat in API_ENDPOINTS:
            fetched_data = fetch_category(cat, date_str, stn_id)
            if fetched_data: cache[key][cat] = fetched_data
            time.sleep(0.5)
            
        if idx % 10 == 0: save_cache(cache)
    
    save_cache(cache)

    # 3. 원본 정보 + 기상 정보 결합하여 파일별 저장
    weather_cols = [col for var_map in WEATHER_VARS.values() for col in var_map.values()]

    for fp, data in rows_by_file.items():
        # 출력 파일 이름 지정: _attendance.csv -> _attendance_weather.csv
        output_file = fp.replace(".csv", "_weather.csv")
        
        # 원본 파일의 컬럼명에 날씨 컬럼을 그대로 이어붙임
        original_fieldnames = data["fieldnames"]
        all_cols = original_fieldnames + ["지점번호", "관측도시"] + weather_cols

        with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            
            for row in data["rows"]:
                stadium = row.get("구장", "").strip()
                stn_id = STADIUM_STN_MAP.get(stadium)
                date_str = row.get("경기날짜", row.get("날짜", ""))
                date_api = kbo_date_to_yyyymmdd(date_str)
                
                # 원본 데이터를 그대로 복사한 뒤, 날씨 정보만 추가
                out = row.copy()
                out["지점번호"] = stn_id or ""
                out["관측도시"] = STN_CITY_MAP.get(stn_id, "") if stn_id else ""
                
                if stn_id: out.update(extract_weather(cache.get(f"{date_api}_{stn_id}", {})))
                else: out.update({c: "" for c in weather_cols})
                
                writer.writerow(out)

        log.info(f"완료! 개별 저장되었습니다: {output_file}")

if __name__ == "__main__":
    main()