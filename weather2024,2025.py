"""
KBO 경기 기상 데이터 수집기 v15 (습도 저장 버그 수정 및 데이터 정밀 매핑)
"""

import requests
import json
import csv
import time
import logging
import re
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  ① 경로 및 환경 설정
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KBO_FILES = [
    os.path.join(BASE_DIR, "kbo_2024_attendance.csv"),
    os.path.join(BASE_DIR, "kbo_2025_attendance.csv")
]
CACHE_FILE = os.path.join(BASE_DIR, "weather_cache.json")
AUTH_KEY = "NtKXD3bRQkCSlw920cJAyA"

# [수정] 기상청 API별 실제 데이터 키 매핑 (이미지 분석 결과 반영)
WEATHER_VARS = {
    "기온": {"TA_DAVG": "일평균기온(C)"},        # 특수기호 ° 제외하여 호환성 높임
    "강수": {"RN_DSUM": "일합계강수량(mm)"},
    "풍속": {"WS_DAVG": "일평균풍속(m/s)"},
    "습도": {"RHM_AVG": "일평균상대습도(%)"},    # RHM_DAVG -> RHM_AVG로 수정
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
    "대전": 133, "한밭": 133, "한화": 133, "포항": 138,
    "대구": 143, "삼성": 143, "울산": 152, "창원": 155, "NC": 155,
    "광주": 156, "KIA": 156, "사직": 159, "롯데": 159,
}

STN_CITY_MAP = {108: "서울", 112: "인천", 119: "수원", 133: "대전", 138: "포항", 143: "대구", 152: "울산", 155: "창원", 156: "광주", 159: "부산"}

# ══════════════════════════════════════════════════════════════
#  ② 유틸리티 함수
# ══════════════════════════════════════════════════════════════
def kbo_date_to_yyyymmdd(date_str):
    return re.sub(r"\D", "", str(date_str))

def clean_weather_value(val):
    if val is None: return ""
    v = str(val).strip()
    return "" if v in ("-", "-9", "-9.0", "-9.9", "-99.9", "-999", "-999.0", "None", "nan", "=", "-99") else v

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
#  ③ 메인 로직
# ══════════════════════════════════════════════════════════════
def main():
    # 1. 파일 로드
    files_data = {}
    for fp in KBO_FILES:
        if os.path.exists(fp):
            with open(fp, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                files_data[fp] = {"fieldnames": reader.fieldnames, "rows": list(reader)}
        else: log.error(f"파일 없음: {fp}")

    if not files_data: return

    # 2. 캐시 로드 및 부족한 데이터 수집
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f: cache = json.load(f)

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
        # 습도 데이터가 JSON에 없는 경우에만 새로 API 호출
        if "습도" not in cache[key] or not cache[key]["습도"]:
            log.info(f"  [{idx}/{len(req_list)}] {dt} (지점 {stn}) 습도 수집 중...")
            res = fetch_weather("습도", dt, stn)
            if res: cache[key]["습도"] = res
            time.sleep(0.3)
            # 캐시 즉시 저장
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # 3. CSV 저장 (강력한 매핑 적용)
    weather_cols = [c for v in WEATHER_VARS.values() for c in v.values()]
    for fp, data in files_data.items():
        out_path = fp.replace(".csv", "_weather.csv")
        # 기존 헤더에서 날씨 컬럼이 이미 있다면 중복되지 않게 처리
        base_fields = [f for f in data["fieldnames"] if f not in weather_cols + ["지점번호", "관측도시"]]
        headers = base_fields + ["지점번호", "관측도시"] + weather_cols
        
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in data["rows"]:
                stn = STADIUM_STN_MAP.get(r.get("구장", "").strip())
                dt = kbo_date_to_yyyymmdd(r.get("경기날짜", r.get("날짜", "")))
                row_out = {k: v for k, v in r.items() if k in headers} # 필요한 컬럼만 복사
                row_out.update({"지점번호": stn or "", "관측도시": STN_CITY_MAP.get(stn, "") if stn else ""})
                
                c_info = cache.get(f"{dt}_{stn}", {})
                for cat, v_map in WEATHER_VARS.items():
                    api_data = c_info.get(cat, {})
                    for api_key, out_col in v_map.items():
                        # 1순위: 지정된 키로 검색
                        val = api_data.get(api_key)
                        # 2순위: (유연한 검색) 키 이름에 RHM과 AVG가 들어있는지 확인
                        if val is None:
                            for k in api_data.keys():
                                if "RHM" in k and "AVG" in k:
                                    val = api_data[k]
                                    break
                        row_out[out_col] = clean_weather_value(val)
                writer.writerow(row_out)
        log.info(f"최종 저장 완료: {os.path.basename(out_path)}")

if __name__ == "__main__":
    main()