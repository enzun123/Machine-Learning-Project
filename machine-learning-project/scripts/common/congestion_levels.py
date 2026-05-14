"""혼잡도(%)에 따른 운영 단계·액션 플랜 (Streamlit UI와 분리)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CongestionActionPlan:
    level: str
    action_class: str
    action_title: str
    action_msg: str


def classify_congestion_pct(congestion_pct: float) -> CongestionActionPlan:
    """혼잡도 % → 카드용 레벨·CSS 클래스·액션 플랜."""
    x = float(congestion_pct)
    if x < 50:
        return CongestionActionPlan(
            level="LOW",
            action_class="action-low",
            action_title="🟢 [비용 절감 모드]",
            action_msg=(
                "식자재 발주를 평소 대비 축소하고, "
                "일부 구역 매점 운영을 최소화하여 "
                "인력을 효율적으로 운용하세요."
            ),
        )
    if x < 80:
        return CongestionActionPlan(
            level="NORMAL",
            action_class="action-mid",
            action_title="🟡 [일반 운영 모드]",
            action_msg=(
                "기본 매뉴얼에 따라 운영을 준비하세요. "
                "입장 동선과 매점 운영 상태를 "
                "사전에 점검하세요."
            ),
        )
    return CongestionActionPlan(
        level="HIGH",
        action_class="action-high",
        action_title="🔴 [안전 강화 모드]",
        action_msg=(
            "게이트 주변 안전 요원을 "
            "20% 추가 배치하고, "
            "매점 재고 소진에 대비해 "
            "발주량을 최대로 늘리세요."
        ),
    )
