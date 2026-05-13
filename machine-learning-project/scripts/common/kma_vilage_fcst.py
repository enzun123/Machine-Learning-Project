"""
동네예보 API허브(typ02) — 초단기예보 RN1(1시간 강수 mm) 및 단기예보 POP(강수확률 %) 참고.

- 초단기: ``getUltraSrtFcst`` — 발표 시각 기준 약 6시간 구간, RN1.
  (발표 후보는 **현재 시각**과 **개시 3시간 전 시각**을 앵커로 합쳐, 같은 날 참고 시각에 맞춰 조회합니다.)
- 단기(폴백): ``getVilageFcst`` — 3시간 단위 발표, POP·PCP 등.

인증: ``KMA_APIHUB_AUTH_KEY`` (동일 계정에서 **동네예보** 활용신청 필요).

격자(nx, ny)는 구장 대표값(행정구역 중심 근사). 공식 격자와 다를 수 있습니다.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KST = ZoneInfo("Asia/Seoul")

ULTRA_URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtFcst"
VILAGE_URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"

# weather_api.STADIUM_STN_MAP 권역과 맞춘 대표 격자 (동네예보 격자; 근사)
STADIUM_GRID: dict[str, tuple[int, int]] = {
    "잠실": (62, 126),
    "두산": (62, 126),
    "LG": (62, 126),
    "고척": (58, 125),
    "키움": (58, 125),
    "수원": (61, 121),
    "KT": (61, 121),
    "인천": (55, 124),
    "문학": (55, 124),
    "SSG": (55, 124),
    "사직": (98, 76),
    "롯데": (98, 76),
    "한밭": (67, 100),
    "대전": (67, 100),
    "한화": (67, 100),
    "광주": (58, 74),
    "KIA": (58, 74),
    "대구": (89, 90),
    "삼성": (89, 90),
    "울산": (102, 84),
    "창원": (91, 77),
    "NC": (91, 77),
    "청주": (69, 106),
    "포항": (102, 94),
}


def _auth_key() -> str:
    k = os.environ.get("KMA_APIHUB_AUTH_KEY", "").strip()
    if k:
        return k
    try:
        from data_collection.weather_api import AUTH_KEY

        return str(AUTH_KEY).strip()
    except Exception:
        return ""


def stadium_grid_xy(stadium: str) -> tuple[int, int] | None:
    s = str(stadium).strip()
    if s in STADIUM_GRID:
        return STADIUM_GRID[s]
    for needle, xy in STADIUM_GRID.items():
        if needle in s:
            return xy
    return None


_RE_API_SECRET = re.compile(r"((?:authKey|serviceKey)=)([^&\s#'\"]+)", re.I)


def redact_api_secrets(text: object) -> str:
    """URL·예외 문자열에 포함된 API 인증 쿼리 값 마스킹."""
    if text is None:
        return ""
    return _RE_API_SECRET.sub(r"\1***", str(text))


def _short_net_err(msg: str) -> str:
    m = redact_api_secrets(msg)
    low = m.lower()
    if "timeout" in low or "timed out" in low:
        return (
            "네트워크 오류: 기상청 API허브(apihub.kma.go.kr) 응답이 지연되거나 "
            "연결이 끊어졌습니다. (방화벽·VPN·재시도)"
        )
    return m


def _items_from_body(body: dict) -> list[dict]:
    resp = body.get("response") or {}
    bd = resp.get("body") or {}
    items = bd.get("items")
    if not items:
        return []
    it = items.get("item")
    if it is None:
        return []
    if isinstance(it, list):
        return it
    return [it]


def _api_err_msg(body: dict) -> str | None:
    try:
        h = (body.get("response") or {}).get("header") or {}
        code = str(h.get("resultCode", "")).strip()
        msg = str(h.get("resultMsg", "")).strip()
        if code and code != "00":
            return f"{msg or code} (code={code})"
    except Exception:
        pass
    return None


def _parse_rn1_mm(raw: str) -> float | None:
    t = str(raw).strip()
    if not t or t in ("강수없음", "-", "null"):
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_pop_pct(raw: str) -> float | None:
    m = re.search(r"(\d+)", str(raw).strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _ultra_base_candidates(now_kst: datetime, n: int = 6) -> list[tuple[str, str]]:
    """(base_date yyyymmdd, base_time HHMM) 최신부터 n개, KST."""
    now_kst = now_kst.astimezone(KST) if now_kst.tzinfo else now_kst.replace(tzinfo=KST)
    if now_kst.minute >= 45:
        b = now_kst.replace(minute=30, second=0, microsecond=0)
    else:
        b = (now_kst - timedelta(hours=1)).replace(minute=30, second=0, microsecond=0)
    out: list[tuple[str, str]] = []
    cur = b
    for _ in range(n):
        out.append((cur.strftime("%Y%m%d"), cur.strftime("%H%M")))
        cur = cur - timedelta(hours=1)
    return out


def _merged_ultra_base_candidates(
    now_kst: datetime,
    target_kst: datetime,
    n_each: int = 8,
) -> list[tuple[str, str]]:
    """
    초단기 발표 시각 후보. ``now`` 기준뿐 아니라 **참고 시각(t3)** 근처 앵커도 사용해,
    같은 날 개시·3시간 전이 가까울 때 RN1 적중률을 높입니다.
    미래 발표 시각은 API에 쓸 수 없어 ``now_kst`` 이하만 남깁니다.
    """
    now_kst = now_kst.astimezone(KST) if now_kst.tzinfo else now_kst.replace(tzinfo=KST)
    tgt = target_kst.astimezone(KST) if target_kst.tzinfo else target_kst.replace(tzinfo=KST)
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str]] = []

    def _base_dt(bd: str, bt: str) -> datetime:
        return datetime.strptime(bd + str(bt).zfill(4), "%Y%m%d%H%M").replace(tzinfo=KST)

    for anchor in (now_kst, tgt):
        for bd, bt in _ultra_base_candidates(anchor, n=n_each):
            if (bd, bt) in seen:
                continue
            try:
                bdt = _base_dt(bd, bt)
            except ValueError:
                continue
            if bdt > now_kst:
                continue
            seen.add((bd, bt))
            merged.append((bd, bt))

    merged.sort(key=lambda p: _base_dt(p[0], p[1]), reverse=True)
    return merged


def _vilage_base_candidates(now_kst: datetime, n: int = 8) -> list[tuple[str, str]]:
    """단기예보 발표 시각(02,05,…,23) — 현재(KST) 이전 중 최신순."""
    now_kst = now_kst.astimezone(KST) if now_kst.tzinfo else now_kst.replace(tzinfo=KST)
    slots = [23, 20, 17, 14, 11, 8, 5, 2]
    all_c: list[tuple[datetime, str, str]] = []
    for day_off in range(0, 4):
        d = now_kst.date() - timedelta(days=day_off)
        for h in slots:
            tslot = datetime(d.year, d.month, d.day, h, 0, tzinfo=KST)
            if tslot <= now_kst:
                all_c.append((tslot, d.strftime("%Y%m%d"), f"{h:02d}00"))
    all_c.sort(key=lambda x: x[0], reverse=True)
    return [(bd, bt) for _, bd, bt in all_c[:n]]


def fetch_ultra_srt_fcst(
    nx: int,
    ny: int,
    base_date: str,
    base_time: str,
    auth_key: str,
) -> tuple[list[dict], str | None]:
    if not auth_key.strip():
        return [], "KMA_APIHUB_AUTH_KEY 가 비어 있습니다."
    params = {
        "pageNo": 1,
        "numOfRows": 500,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": int(nx),
        "ny": int(ny),
        "authKey": auth_key.strip(),
    }
    try:
        r = requests.get(ULTRA_URL, params=params, timeout=30)
    except Exception as e:
        return [], _short_net_err(f"네트워크 오류: {e}")
    try:
        body = r.json()
    except Exception:
        return [], redact_api_secrets(f"JSON 파싱 실패 HTTP {r.status_code}: {r.text[:200]!r}")
    err = _api_err_msg(body)
    if err:
        return [], err
    return _items_from_body(body), None


def fetch_vilage_fcst(
    nx: int,
    ny: int,
    base_date: str,
    base_time: str,
    auth_key: str,
) -> tuple[list[dict], str | None]:
    if not auth_key.strip():
        return [], "KMA_APIHUB_AUTH_KEY 가 비어 있습니다."
    params = {
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": int(nx),
        "ny": int(ny),
        "authKey": auth_key.strip(),
    }
    try:
        r = requests.get(VILAGE_URL, params=params, timeout=35)
    except Exception as e:
        return [], _short_net_err(f"네트워크 오류: {e}")
    try:
        body = r.json()
    except Exception:
        return [], redact_api_secrets(f"JSON 파싱 실패 HTTP {r.status_code}: {r.text[:200]!r}")
    err = _api_err_msg(body)
    if err:
        return [], err
    return _items_from_body(body), None


def _rn1_at_fcst_hour(items: list[dict], fcst_date: str, fcst_hour: str) -> float | None:
    """fcst_hour: '1500' 형태. 해당 시각 RN1(mm)."""
    want_t = fcst_hour.zfill(4)
    for it in items:
        if str(it.get("category", "")).upper() != "RN1":
            continue
        if str(it.get("fcstDate", "")) != fcst_date:
            continue
        ft = str(it.get("fcstTime", "")).zfill(4)
        if ft == want_t:
            v = _parse_rn1_mm(str(it.get("fcstValue", "")))
            if v is not None:
                return v
    return None


def _rn1_nearest_to_target(items: list[dict], tgt: pd.Timestamp, max_sec: int = 7200) -> float | None:
    """같은 응답에서 ``tgt``(KST)에 가장 가까운 정시 RN1."""
    tgt = tgt.tz_convert(KST) if tgt.tzinfo else tgt.tz_localize(KST)
    t_py = tgt.to_pydatetime()
    best_v: float | None = None
    best_d: float | None = None
    for it in items:
        if str(it.get("category", "")).upper() != "RN1":
            continue
        fd = str(it.get("fcstDate", ""))
        ft = str(it.get("fcstTime", "")).zfill(4)
        if len(fd) != 8 or len(ft) != 4:
            continue
        try:
            dt = datetime.strptime(fd + ft, "%Y%m%d%H%M").replace(tzinfo=KST)
        except ValueError:
            continue
        v = _parse_rn1_mm(str(it.get("fcstValue", "")))
        if v is None:
            continue
        d = abs((dt - t_py).total_seconds())
        if d <= max_sec and (best_d is None or d < best_d):
            best_d = d
            best_v = v
    return best_v


def ultra_rn1_mm_for_target_hour(
    nx: int,
    ny: int,
    target_kst: pd.Timestamp,
    auth_key: str | None = None,
) -> tuple[float | None, str | None, str | None]:
    """
    초단기예보 RN1(1시간 강수) 중 ``target_kst`` 정시에 가장 가까운 시각 값.

    반환: (mm 또는 None, 사용한 base_date+base_time 요약, 오류)
    """
    auth = (auth_key or _auth_key()).strip()
    if not auth:
        return None, None, "KMA_APIHUB_AUTH_KEY 가 없습니다."

    tgt = pd.Timestamp(target_kst)
    if tgt.tzinfo is None:
        tgt = tgt.tz_localize(KST)
    else:
        tgt = tgt.tz_convert(KST)

    fcst_date = tgt.strftime("%Y%m%d")
    fcst_time = tgt.floor("h").strftime("%H%M")

    now = datetime.now(KST)
    last_err = None
    for bd, bt in _merged_ultra_base_candidates(now, tgt.to_pydatetime(), n_each=8):
        items, err = fetch_ultra_srt_fcst(nx, ny, bd, bt, auth)
        if err:
            last_err = err
            continue
        if not items:
            last_err = "초단기 응답 항목 없음"
            continue
        v = _rn1_at_fcst_hour(items, fcst_date, fcst_time)
        if v is None:
            v = _rn1_nearest_to_target(items, tgt)
        if v is not None:
            return v, f"{bd}/{bt}", None
        last_err = "해당 시각 RN1 없음"
    return None, None, last_err or "초단기예보 조회 실패"


def vilage_pop_at_nearest_fcst(
    nx: int,
    ny: int,
    target_kst: pd.Timestamp,
    auth_key: str | None = None,
) -> tuple[float | None, str | None, str | None]:
    """단기예보 POP(%) — ``target``에 가장 가까운 fcstDate+fcstTime."""
    auth = (auth_key or _auth_key()).strip()
    if not auth:
        return None, None, "KMA_APIHUB_AUTH_KEY 가 없습니다."
    tgt = pd.Timestamp(target_kst)
    if tgt.tzinfo is None:
        tgt = tgt.tz_localize(KST)
    else:
        tgt = tgt.tz_convert(KST)

    best_pop: float | None = None
    best_dt: datetime | None = None
    base_used = None
    last_err = None

    for bd, bt in _vilage_base_candidates(datetime.now(KST), n=6):
        items, err = fetch_vilage_fcst(nx, ny, bd, bt, auth)
        if err:
            last_err = err
            continue
        base_used = f"{bd}/{bt}"
        for it in items:
            if str(it.get("category", "")).upper() != "POP":
                continue
            fd = str(it.get("fcstDate", ""))
            ft = str(it.get("fcstTime", "")).zfill(4)
            if len(fd) != 8 or len(ft) != 4:
                continue
            try:
                dt = datetime.strptime(fd + ft, "%Y%m%d%H%M").replace(tzinfo=KST)
            except ValueError:
                continue
            pop = _parse_pop_pct(str(it.get("fcstValue", "")))
            if pop is None:
                continue
            if best_dt is None or abs((dt - tgt).total_seconds()) < abs((best_dt - tgt).total_seconds()):
                best_dt = dt
                best_pop = pop
        if best_pop is not None:
            return best_pop, base_used, None
    return None, None, last_err or "단기예보 POP 없음"


def three_hours_before_game_start(game_start: pd.Timestamp) -> pd.Timestamp:
    """개시 시각(KST) 기준 3시간 전 시각."""
    t = pd.Timestamp(game_start)
    if t.tzinfo is None:
        t = t.tz_localize(KST)
    else:
        t = t.tz_convert(KST)
    return t - pd.Timedelta(hours=3)


def forecast_ref_for_rain_cancel_rules(
    stadium: str,
    game_start: pd.Timestamp,
    auth_key: str | None = None,
) -> dict:
    """
    규칙 기반 참고용 요약 (우천 **취소 예측 아님**).

    - 가능하면 초단기 RN1(개시 3시간 전 정시 근처).
    - 실패 시 단기 POP 폴백.
    """
    auth = (auth_key or _auth_key()).strip()
    if not auth:
        return {
            "ok": False,
            "msg": "KMA_APIHUB_AUTH_KEY 가 없습니다. 환경변수 또는 `.streamlit/secrets.toml`을 설정하세요.",
            "mode": "none",
        }

    grid = stadium_grid_xy(stadium)
    if grid is None:
        return {"ok": False, "msg": "구장에 매핑된 동네예보 격자가 없습니다.", "mode": "none"}

    nx, ny = grid
    t3 = three_hours_before_game_start(game_start)
    now = datetime.now(KST)

    if t3 > now + timedelta(days=10):
        return {
            "ok": False,
            "msg": "개시 3시간 전이 너무 먼 미래(약 10일 초과)라, 이 화면의 단기 조회만으로는 참고하기 어렵습니다.",
            "mode": "none",
            "target_kst": str(t3),
        }

    if t3 + timedelta(hours=1) < now - timedelta(days=1):
        return {
            "ok": False,
            "msg": "개시 3시간 전이 과거(1일 이상 전)입니다. 과거는 ASOS 실황·기록을 참고하세요.",
            "mode": "none",
            "target_kst": str(t3),
        }

    rn1, ultra_base, err_u = ultra_rn1_mm_for_target_hour(nx, ny, t3, auth_key=auth)
    if rn1 is not None:
        return {
            "ok": True,
            "mode": "ultra_rn1",
            "mm_h": float(rn1),
            "base": ultra_base,
            "target_kst": str(t3),
            "nx": nx,
            "ny": ny,
            "detail": None,
        }

    pop, vbase, err_v = vilage_pop_at_nearest_fcst(nx, ny, t3, auth_key=auth)
    if pop is not None:
        return {
            "ok": True,
            "mode": "vilage_pop",
            "pop_pct": float(pop),
            "base": vbase,
            "target_kst": str(t3),
            "nx": nx,
            "ny": ny,
            "detail": err_u,
        }

    raw = " | ".join(redact_api_secrets(x) for x in (err_u, err_v) if x)
    msg = (
        "현재 시각 기준으로 동네예보(초단기·단기)에서 개시 3시간 전 참고값을 가져오지 못했습니다. "
        "경기 당일 또는 개시에 가까운 시점에 다시 확인해 보세요."
    )
    return {
        "ok": False,
        "msg": msg,
        "detail": raw or None,
        "mode": "none",
        "target_kst": str(t3),
    }


def rule_band_from_mm_h(mm: float) -> tuple[str, str, str]:
    """(제목, 본문, css low|mid|high) — KBO 3시간 전 10mm/h 검토선 참고."""
    x = float(mm)
    if x <= 0:
        return (
            "예보 1시간 강수 거의 없음 (참고)",
            "보도에서 말하는 ‘개시 3시간 전 시간당 10mm 예보’ 검토선보다 훨씬 낮은 구간입니다.",
            "low",
        )
    if x < 5:
        return (
            "5mm/h 미만 (참고)",
            "1시간 전 실강우 5mm/h 검토선·3시간 전 10mm/h 예보 검토선보다 낮은 편입니다.",
            "low",
        )
    if x < 10:
        return (
            "5~10mm/h 부근 (참고)",
            "우천·연기 이슈가 거론될 수 있는 구간으로 **보도에서 종종 인용**됩니다. 실제는 예보·구장·심판 판단입니다.",
            "mid",
        )
    return (
        "10mm/h 이상에 근접·이상 (참고)",
        "많은 **보도 요약**에서 ‘개시 3시간 전 시간당 10mm 이상 **예보**’ 시 검토된다고 전해집니다. **절대 기준은 아닙니다.**",
        "high",
    )


def rule_band_from_pop(pop_pct: float) -> tuple[str, str, str]:
    p = float(pop_pct)
    if p < 40:
        return (
            f"강수확률 {p:.0f}% (단기예보, 참고)",
            "초단기 1시간 강수(RN1)를 가져오지 못해 **강수확률**만 표시합니다. 10mm/h 검토선과 직접 대응하지는 않습니다.",
            "low",
        )
    if p < 70:
        return (
            f"강수확률 {p:.0f}% (단기예보, 참고)",
            "비 가능성이 있는 편입니다. **시간당 강수(mm/h)** 는 기상청 상세예보·통보문을 확인하세요.",
            "mid",
        )
    return (
        f"강수확률 {p:.0f}% (단기예보, 참고)",
        "비 가능성이 높게 나온 편입니다. 취소·연기 여부는 리그·구장 운영과 별개로 판단됩니다.",
        "high",
    )


_DISC_RAINOUT = (
    "취소·연기 여부는 KBO·구장·심판·실제 기상에 따라 결정되며, "
    "위 문구는 예보 수치를 바탕으로 한 비공식 참고이며 확정 예측이 아닙니다."
)


def rainout_cancel_guidance(ref: dict) -> dict[str, object]:
    """
    우천 취소·연기 **참고** 문구 (개시 3시간 전 예보 기준, 확정 아님).

    ``ref``는 ``forecast_ref_for_rain_cancel_rules`` 반환과 동일한 키를 기대합니다.

    반환: ``band`` (low|mid|high|warn), ``headline`` (str), ``lines`` (list[str]), ``source`` (str)
    """
    if not ref.get("ok"):
        lines: list[str] = [
            "조회 실패 사유는 바로 위 ‘동네예보 참고’ 박스 괄호 안에만 적어 두었습니다.",
            _DISC_RAINOUT,
        ]
        return {
            "band": "warn",
            "headline": "우천 취소·연기 참고: 예보 없음",
            "lines": lines,
            "source": "",
        }

    mode = ref.get("mode")
    if mode == "ultra_rn1":
        mm = float(ref["mm_h"])
        _, body_raw, band = rule_band_from_mm_h(mm)
        body_plain = str(body_raw).replace("**", "")
        if band == "low":
            headline = "취소·연기 검토가 보도·언론에 거론될 가능성: 낮은 편(예보 기준)"
        elif band == "mid":
            headline = "취소·연기 검토가 거론될 수 있는 예보 구간(참고)"
        else:
            headline = "보도에서 자주 인용하는 ‘개시 3시간 전’ 강우 예보 검토선에 근접·이상(참고)"
        lines = [
            f"개시 3시간 전에 가까운 초단기 1시간 강수(RN1)는 약 {mm:g} mm/h 입니다.",
            body_plain,
            _DISC_RAINOUT,
        ]
        return {
            "band": band,
            "headline": headline,
            "lines": lines,
            "source": "기상청 API허브 동네예보 초단기 RN1",
        }

    if mode == "vilage_pop":
        pop = float(ref["pop_pct"])
        _, body_raw, band = rule_band_from_pop(pop)
        body_plain = str(body_raw).replace("**", "")
        if band == "low":
            headline = "강수확률이 낮아, 우천 취소·연기 언급 가능성은 제한적(참고)"
        elif band == "mid":
            headline = "비 가능성이 있는 편 — 취소·연기는 운영·심판 판단(참고)"
        else:
            headline = "비 가능성이 높게 보임 — 실제 취소·연기는 별도 판단(참고)"
        lines = [
            f"초단기 RN1 대신 단기 강수확률(POP) 약 {pop:.0f}% 만 사용했습니다.",
            body_plain,
            _DISC_RAINOUT,
        ]
        return {
            "band": band,
            "headline": headline,
            "lines": lines,
            "source": "기상청 API허브 동네예보 단기 POP",
        }

    lines = ["예보 응답 형식을 해석하지 못했습니다.", _DISC_RAINOUT]
    return {
        "band": "warn",
        "headline": "우천 취소·연기 가능성: 참고 불가",
        "lines": lines,
        "source": "—",
    }
