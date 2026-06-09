"""
RiskMitigationPlanner（风险缓释建议生成器）

Generates risk mitigation actions and monitoring indicators.
Rule-based — no LLM.

ALL output uses risk management language:
  "可考虑" "建议关注" "持续监测" "复核"
NEVER uses trading language:
  "买入" "卖出" "加仓" "减仓" "满仓" "空仓"
"""

from __future__ import annotations

_FORBIDDEN_TERMS = {"买入", "卖出", "加仓", "减仓", "满仓", "空仓",
                     "抄底", "追涨", "杀跌", "推荐", "强烈建议"}


def plan(*, concentration_result: dict | None = None,
         preference_result: dict | None = None,
         liquidity_result: dict | None = None,
         financial_quality_result: dict | None = None,
         **kwargs) -> dict:
    """Generate risk mitigation actions and monitoring indicators.

    Args:
        concentration_result: Output from ConcentrationRiskChecker.
        preference_result: Output from RiskPreferenceMatcher.
        liquidity_result: Output from LiquidityRiskAssessor.
        financial_quality_result: Output from FinancialQualityRiskDetector.

    Returns:
        dict with mitigation_actions, monitoring_indicators.
    """
    actions: list[str] = []
    indicators: list[str] = []

    # ── Concentration mitigation ───────────────────────────────
    conc = concentration_result or {}
    if conc.get("concentration_level") in ("high", "moderate"):
        for issue in conc.get("issues", []):
            if "50%" in issue:
                actions.append(
                    "可考虑将单一类别占比降至50%以下，通过跨资产、跨行业分散配置降低集中度风险。"
                )
            if "权益类合计" in issue:
                actions.append(
                    "建议关注权益类仓位控制，可考虑适当增加固定收益类资产以降低组合波动。"
                )
            if "现金" in issue and ("不足" in issue or "低" in issue):
                actions.append(
                    "建议关注流动性储备，可考虑保留一定比例的现金及货币类资产以应对应急需求。"
                )
        indicators.append("单一类别最大占比（目标<50%）")
        indicators.append("权益类合计占比")

    # ── Preference mismatch mitigation ──────────────────────────
    pref = preference_result or {}
    if pref.get("match_status") == "mismatch":
        for reason in pref.get("mismatch_reasons", []):
            if "权益" in reason:
                actions.append(
                    "当前组合风险已超出用户风险偏好上限，建议关注组合再平衡，"
                    "可考虑将权益类占比调整至风险偏好允许的范围内。"
                )
                band = pref.get("allowed_risk_band", {})
                if band:
                    actions.append(
                        f"参考风险偏好上限：权益类≤{band.get('max_equity', 'N/A')}%，"
                        f"高风险资产合计≤{band.get('max_risky', 'N/A')}%。"
                    )
        indicators.append("组合风险是否在风险偏好允许范围内")

    # ── Liquidity mitigation ───────────────────────────────────
    liq = liquidity_result or {}
    if liq.get("liquidity_level") in ("high_risk", "moderate_risk", "elevated"):
        for finding in liq.get("liquidity_findings", []):
            if "现金流" in finding:
                actions.append(
                    "经营现金流持续为负需重点关注，建议持续监测未来3-6个季度的现金流变化，"
                    "关注企业融资能力和债务到期结构。"
                )
        indicators.append("现金类占比（建议≥5%）")
        indicators.append("经营现金流趋势")

    # ── Financial quality mitigation ───────────────────────────
    fq = financial_quality_result or {}
    for flag in fq.get("risk_flags", []):
        if "经营现金流" in flag:
            actions.append(
                "建议关注利润质量，持续监测经营现金流/净利润比率，"
                "若比率持续偏低，需进一步分析应收账款质量和收入确认政策。"
            )
        if "资产负债率" in flag:
            actions.append(
                "建议关注资产负债率变化趋势，若持续上升，需评估再融资能力和利息覆盖倍数。"
            )
        if "商誉" in flag:
            actions.append(
                "建议关注被并购标的业绩完成情况，定期复核是否需要计提商誉减值准备。"
            )
    indicators.append("资产负债率月度监测")
    indicators.append("经营现金流/净利润比率")

    # ── General indicator ──────────────────────────────────────
    if not indicators:
        indicators.append("组合定期再平衡（建议每季度复核一次）")
        indicators.append("关注市场系统性风险事件")

    # ── Safety check: no trading language ──────────────────────
    for term in _FORBIDDEN_TERMS:
        for action in actions:
            if term in action:
                raise RuntimeError(
                    f"Forbidden term '{term}' found in mitigation action: {action}"
                )

    return {
        "mitigation_actions": actions,
        "monitoring_indicators": indicators,
    }
