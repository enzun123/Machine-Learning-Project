import requests
import json
import csv
import time
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  ① 경로 및 API 환경 설정 (pathlib 적용)
# ══════════════════════════════════════════════════════════════
# 현재 파이썬 파일 위치를 기준으로 프로젝트 경로를 계산합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# 파일 경로들을 Path 객체로 안전하게 생성
KBO_FILES = [
    RAW_DIR / "kbo_2024_attendance.csv",
    RAW_DIR / "kbo_2025_attendance.csv"
]
CACHE_FILE = EXTERNAL_DIR / "weather_cache.json"

AUTH_KEY = "NtKXD3bRQkCSlw920cJAyA"

# CSV에 저장될 컬럼명 및 API 매핑
WEATHER_VARS = {
    "기온": {"TA_DAVG": "일평균기온(°C)"},
    "강수": {"RN_DSUM": "일합계강수량(mm)"},
    "풍속": {"WS_DAVG": "일평균풍속(m/s)"},
    "습도": {"RHM_AVG": "일평균상대습도(%)"},
}

API_ENDPOINTS = {
    "기온": "https://apihub.kma.go.kr/api/typ01/url/sts_ta.php",
    "강수": "https://apihub.kma.go.kr/api/typ01/url/sts_rn.php",
    "풍속": "https://apihub.kma.go.kr/api/typ01/url/sts_wind.php",
    "습도": "https://apihub.kma.go.kr/api/typ01/url/sts_rhm.php",
}

STADIUM_STN_MAP = {
    "잠실": 108, "두산": 108, "LG": 108, "고척": 108, "키움": 108,
    "문학": 112, "SSG": 112, "수원": 119, "KT": 119,
    "청주": 131, "대전": 133, "한밭": 133, "한화": 133, "포항": 138,
    "대구": 143, "삼성": 143, "울산": 152, "창원": 155, "NC": 155,
    "광주": 156, "KIA": 156, "사직": 159, "롯데": 159,
}

STN_CITY_MAP = {108: "서울", 112: "인천", 119: "수원", 131: "청주", 133: "대전", 138: "포항", 143: "대구", 152: "울산", 155: "창원", 156: "광주", 159: "부산"}

# ══════════════════════════════════════════════════════════════
#  ② 데이터 처리 보조 함수
# ══════════════════════════════════════════════════════════════
def kbo_date_to_yyyymmdd(date_str):
    if not date_str: return ""
    return re.sub(r"\D", "", str(date_str))

def clean_weather_value(val):
    if val is None: return ""
    v = str(val).strip()
    if v in ("-", "-9", "-9.0", "-9.9", "-99.9", "-999", "-999.0", "None", "nan", "=", "-99"):
        return ""
    return v


def _extract_fallback_value(api_data, category):
    """카테고리별 유사 키를 탐색해 값을 보완 추출."""
    if not api_data:
        return None
    for k, v in api_data.items():
        if category == "습도" and "RHM" in k and "AVG" in k:
            return v
        if category == "기온" and "TA" in k and "AVG" in k:
            return v
        if category == "풍속" and "WS" in k and "AVG" in k:
            return v
        if category == "강수" and "RN" in k and "SUM" in k:
            return v
    return None


def impute_humidity_from_neighbors(cache, date_str, stn_id, max_days=7):
    """
    습도 결측 시 같은 지점의 전/후 날짜 값을 탐색해 보간.
    - 양쪽 값이 있으면 평균
    - 한쪽만 있으면 해당 값 사용
    """
    if not date_str or not stn_id:
        return ""

    try:
        base = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return ""

    prev_val, next_val = None, None

    for offset in range(1, max_days + 1):
        prev_dt = (base - timedelta(days=offset)).strftime("%Y%m%d")
        next_dt = (base + timedelta(days=offset)).strftime("%Y%m%d")

        if prev_val is None:
            prev_info = cache.get(f"{prev_dt}_{stn_id}", {}).get("습도", {})
            raw = _extract_fallback_value(prev_info, "습도")
            cleaned = clean_weather_value(raw)
            if cleaned != "":
                try:
                    prev_val = float(cleaned)
                except ValueError:
                    pass

        if next_val is None:
            next_info = cache.get(f"{next_dt}_{stn_id}", {}).get("습도", {})
            raw = _extract_fallback_value(next_info, "습도")
            cleaned = clean_weather_value(raw)
            if cleaned != "":
                try:
                    next_val = float(cleaned)
                except ValueError:
                    pass

        if prev_val is not None and next_val is not None:
            break

    if prev_val is not None and next_val is not None:
        return f"{(prev_val + next_val) / 2:.1f}"
    if prev_val is not None:
        return f"{prev_val:.1f}"
    if next_val is not None:
        return f"{next_val:.1f}"
    return ""

def parse_kma_text(raw_text):
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    header, data = None, None
    for line in lines:
        if line.startswith("#"):
            clean = line.lstrip("#").strip()
            if clean and not re.match(r'^[-=]+$', clean) and "START" not in clean: header = clean
        else: data = line; break
    if not header or not data: return {}
    headers = [x for x in header.replace(",", " ").replace("=", " ").split() if x]
    values = [x for x in data.replace(",", " ").replace("=", " ").split() if x]
    return dict(zip(headers, values)) if len(headers) >= 3 and len(values) >= 3 else {}

def fetch_weather(category, date_str, stn_id):
    params = {"tm1": date_str, "tm2": date_str, "stn_id": stn_id, "disp": 1, "help": 1, "authKey": AUTH_KEY}
    try:
        resp = requests.get(API_ENDPOINTS[category], params=params, timeout=20)
        return parse_kma_text(resp.text)
    except: return {}

# ══════════════════════════════════════════════════════════════
#  ③ 메인 실행 로직
# ══════════════════════════════════════════════════════════════
def main():
    log.info(f"프로젝트 루트 확인: {PROJECT_ROOT}")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 원본 파일 로드
    files_data = {}
    for fp in KBO_FILES:
        if fp.exists():
            with open(fp, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                files_data[fp] = {"fieldnames": reader.fieldnames, "rows": list(reader)}
                log.info(f"로드 성공: {fp.name}")
        else:
            log.warning(f"파일을 찾을 수 없음: {fp.name}")

    if not files_data:
        log.error("처리할 수 있는 CSV 파일이 없습니다.")
        return

    # 2. 캐시 로드
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except:
            log.warning("캐시 파일 읽기 실패. 새로 생성합니다.")

    # 3. 데이터 수집
    unique_pairs = set()
    for data in files_data.values():
        for r in data["rows"]:
            stn = STADIUM_STN_MAP.get(r.get("구장", "").strip())
            dt = kbo_date_to_yyyymmdd(r.get("경기날짜", r.get("날짜", "")))
            if stn and dt: unique_pairs.add((dt, stn))

    req_list = sorted(list(unique_pairs))
    for idx, (dt, stn) in enumerate(req_list, 1):
        key = f"{dt}_{stn}"
        if key not in cache: cache[key] = {}
        
        missing_cats = [cat for cat in API_ENDPOINTS if not cache[key].get(cat)]
        if missing_cats:
            log.info(f"  [{idx}/{len(req_list)}] {dt} (지점 {stn}) 수집: {missing_cats}")
            for cat in missing_cats:
                res = fetch_weather(cat, dt, stn)
                if res: cache[key][cat] = res
                time.sleep(0.3)
            
            # 캐시 안전하게 저장
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # 4. CSV 병합 저장
    weather_cols = [c for v in WEATHER_VARS.values() for c in v.values()]
    for fp, data in files_data.items():
        out_path = INTERIM_DIR / fp.name.replace(".csv", "_weather.csv")
        
        base_fields = [f for f in data["fieldnames"] if f not in weather_cols + ["지점번호", "관측도시"]]
        final_headers = base_fields + ["지점번호", "관측도시"] + weather_cols
        
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=final_headers)
            writer.writeheader()
            
            for r in data["rows"]:
                stn = STADIUM_STN_MAP.get(r.get("구장", "").strip())
                dt = kbo_date_to_yyyymmdd(r.get("경기날짜", r.get("날짜", "")))
                
                row_out = {k: v for k, v in r.items() if k in final_headers}
                row_out.update({"지점번호": stn or "", "관측도시": STN_CITY_MAP.get(stn, "") if stn else ""})
                
                c_info = cache.get(f"{dt}_{stn}", {})
                for cat, v_map in WEATHER_VARS.items():
                    api_data = c_info.get(cat, {})
                    for api_key, out_col in v_map.items():
                        val = api_data.get(api_key)
                        if val is None: # 유연한 키 검색
                            val = _extract_fallback_value(api_data, cat)
                        cleaned_val = clean_weather_value(val)
                        # 수원 2건처럼 원천 습도 누락인 경우 전/후 날짜로 보간
                        if cat == "습도" and cleaned_val == "":
                            cleaned_val = impute_humidity_from_neighbors(cache, dt, stn)
                        row_out[out_col] = cleaned_val
                writer.writerow(row_out)
        
        log.info(f"완료되었습니다: {out_path.name}")

if __name__ == "__main__":
    main()