"""
AnomalyRiskDetector（异常风险识别器）

Detects financial anomalies and risk signals from metrics and text.
Rule-based — no LLM.
All output uses risk-warning language, never investment advice.
"""

from __future__ import annotations


def detect_risks(
    *,
    metrics: dict,
    financial_text: str = "",
    **kwargs,
) -> dict:
    """Detect financial anomalies and risk flags.

    Args:
        metrics: dict from FinancialMetricExtractor.
        financial_text: Original financial report text.

    Returns:
        dict with risk_flags (list of risk descriptions).
    """
    risk_flags: list[str] = []

    net_profit = metrics.get("net_profit")
    operating_cash_flow = metrics.get("operating_cash_flow")
    accounts_receivable = metrics.get("accounts_receivable")
    inventory = metrics.get("inventory")
    goodwill = metrics.get("goodwill")
    debt_ratio = metrics.get("debt_ratio")
    revenue = metrics.get("revenue")

    # ── Check 1: Profit growth but negative OCF ───────────────
    if (
        net_profit is not None
        and operating_cash_flow is not None
        and net_profit > 0
        and operating_cash_flow < 0
    ):
        risk_flags.append(
            "⚠ 净利润为正但经营现金流为负：利润质量存在隐忧，"
            "可能存在应收账款大幅增加、收入确认偏激进等问题，"
            "建议关注现金流与利润的背离原因。"
        )

    # ── Check 2: Accounts receivable spike ────────────────────
    if (
        accounts_receivable is not None
        and revenue is not None
        and revenue > 0
    ):
        ar_ratio = accounts_receivable / revenue
        if ar_ratio > 0.60:
            risk_flags.append(
                f"⚠ 应收账款占营收比例高达{ar_ratio:.1%}："
                "回款风险较高，需关注账龄结构及坏账计提是否充分。"
            )

    # ── Check 3: High inventory ───────────────────────────────
    if (
        inventory is not None
        and revenue is not None
        and revenue > 0
    ):
        inv_ratio = inventory / revenue
        if inv_ratio > 0.60:
            risk_flags.append(
                f"⚠ 存货占营收比例高达{inv_ratio:.1%}："
                "存货积压风险，需关注存货跌价准备及行业景气度变化。"
            )

    # ── Check 4: Goodwill impairment risk ─────────────────────
    if (
        goodwill is not None
        and net_profit is not None
        and net_profit > 0
        and goodwill > net_profit * 3
    ):
        risk_flags.append(
            "⚠ 商誉占净利润倍数较高：若被并购标的业绩不及预期，"
            "存在商誉减值风险，将直接影响当期利润。"
        )

    # ── Check 5: High debt ratio + low OCF ────────────────────
    if (
        debt_ratio is not None
        and debt_ratio > 0.70
        and operating_cash_flow is not None
        and operating_cash_flow < 0
    ):
        risk_flags.append(
            "⚠ 高负债率+经营现金流为负：企业偿债能力面临双重压力，"
            "需关注再融资能力和债务到期结构。"
        )

    # ── Check 6: Text keyword detection ───────────────────────
    text = financial_text or ""
    keyword_risks = {
        "亏损": "文本提及亏损，需关注盈利能力的持续性",
        "退市": "文本提及退市风险，需高度关注",
        "诉讼": "文本提及诉讼事项，需评估潜在影响",
        "担保": "文本提及对外担保，需关注或有负债风险",
        "重组": "文本提及重组事项，结果存在不确定性",
    }
    for kw, desc in keyword_risks.items():
        if kw in text:
            risk_flags.append(f"⚠ {desc}")

    return {
        "risk_flags": risk_flags,
        "risk_count": len(risk_flags),
    }
