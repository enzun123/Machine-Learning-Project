# 기상 API 권한·POP 폴백 보강 (`fix/ui`)

**브랜치:** `fix/ui`

기상청 **API허브 동네예보(typ02)** 호출 시 **인증·이용신청(권한) 오류**를 본문 `response.header`가 아니라 **`result.status` / `message` 봉투**로 받는 경우를 처리하고, 단기 POP이 **개시 3시간 전(`t3`)** 에 맞지 않을 때 **경기일 12:00(KST)** 근처 예보 시각으로 한 번 더 조회합니다. Streamlit **기상 디버그** JSON에 `pop_anchor_kst` 필드를 노출합니다.

## 상위 브랜치

- **`feat/2026-schedule-and-weather`** — 동네예보·우천 참고 UI, `kma_vilage_fcst.py` 도입 등  
- **`fix/ui`** — 위 커밋 이후 **API 권한 메시지·POP 앵커**만 추가 수정 (`df6f39b fix : api 권한 문제 해결`)

## 변경 파일

| 파일 | 내용 |
|------|------|
| `scripts/common/kma_vilage_fcst.py` | `_apihub_result_envelope_err`로 typ02 봉투형 오류 파싱, `활용신청` 문구 감지 시 사용자 안내 메시지, `_join_fcst_errors`, 단기 후보 `n=10` 확대, `game_day_noon_kst` + POP 재시도, `forecast_ref_for_rain_cancel_rules`에 `pop_anchor_kst`, 우천 참고 문구 보강 |
| `scripts/app/streamlit_app.py` | `STREAMLIT_DEBUG_WEATHER=1`일 때 디버그 JSON에 **`pop_anchor_kst`** 키 포함 |

## 동작 요약

1. **401 / 활용신청**류 응답을 `response.header`만이 아니라 **`body["result"]`** 에서도 읽음.  
2. 실패 메시지 여러 단계를 **`_join_fcst_errors`** 로 합치고, **`활용신청`** 포함 시 **동네예보 API 승인·`KMA_APIHUB_AUTH_KEY`** 확인 안내로 분기.  
3. 단기 POP: `t3` 실패 시 **경기일 12:00 KST** 근처로 `vilage_pop_at_nearest_fcst` 재시도 → 성공 시 `pop_anchor_kst`와 `detail`에 이유 기록.  
4. `rainout_cancel_guidance`에서 POP만 쓴 경우 **낮 12:00 앵커** 사용 여부를 한 줄로 표시.

## 환경 변수

```bash
export KMA_APIHUB_AUTH_KEY="API허브_인증키"
```

- 마이페이지에서 **`getUltraSrtFcst`**, **`getVilageFcst`**(동네예보)에 해당 키가 승인되어 있는지 확인.  
- Streamlit **Secrets**에 넣는 경우도 동일 변수명으로 맞추면 됩니다.

## 확인 방법

```bash
cd machine-learning-project
export STREAMLIT_DEBUG_WEATHER=1
export KMA_APIHUB_AUTH_KEY=...
streamlit run scripts/app/streamlit_app.py
```

사이드바·예보 참고 흐름에서 오류 시 메시지가 **권한(활용신청)** 인지 **일반 네트워크/시각** 문제인지 구분되는지 확인합니다.

## 문의

- **팀장:** 허은준  
- **연락:** enzun123@gmail.com  
