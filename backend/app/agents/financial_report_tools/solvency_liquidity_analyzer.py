"""
SolvencyLiquidityAnalyzer（偿债与流动性分析器）

Analyzes solvency and liquidity based on extracted metrics.
Rule-based — no LLM.
"""

from __future__ import annotations


def analyze_solvency(
    *,
    metrics: dict,
    **kwargs,
) -> dict:
    """Analyze solvency and liquidity from financial metrics.

    Args:
        metrics: dict from FinancialMetricExtractor.

    Returns:
        dict with solvency_level, liquidity_findings, debt_risks.
    """
    findings: list[str] = []
    risks: list[str] = []

    debt_ratio = metrics.get("debt_ratio")
    operating_cash_flow = metrics.get("operating_cash_flow")
    net_profit = metrics.get("net_profit")

    score = 0
    data_points = 0

    # ── Debt ratio analysis ───────────────────────────────────
    if debt_ratio is not None:
        data_points += 1
        if debt_ratio <= 0.40:
            findings.append(f"资产负债率{debt_ratio:.1%}，财务结构稳健")
            score += 2
        elif debt_ratio <= 0.60:
            findings.append(f"资产负债率{debt_ratio:.1%}，处于合理范围")
            score += 1
        elif debt_ratio <= 0.80:
            findings.append(f"资产负债率{debt_ratio:.1%}，负债水平偏高")
            risks.append(f"资产负债率{debt_ratio:.1%}超过60%，需关注偿债压力")
            score -= 1
        else:
            findings.append(f"资产负债率{debt_ratio:.1%}，处于高位，财务风险较大")
            risks.append(f"资产负债率{debt_ratio:.1%}超过80%，存在较高偿债风险")
            score -= 2

    # ── Operating cash flow analysis ──────────────────────────
    if operating_cash_flow is not None:
        data_points += 1
        if operating_cash_flow > 0:
            findings.append("经营活动现金流为正，日常经营造血能力正常")
            score += 1
        else:
            findings.append("经营活动现金流为负，需关注现金流质量")
            risks.append("经营活动现金流为负，企业经营造血能力不足，可能面临流动性压力")
            score -= 1

    # ── Cash flow vs profit ───────────────────────────────────
    if operating_cash_flow is not None and net_profit is not None:
        data_points += 1
        if net_profit > 0 and operating_cash_flow < 0:
            risks.append(
                "净利润为正但经营现金流为负，利润质量可能偏低，"
                "存在应收账款积压或收入确认偏激进的迹象"
            )
            score -= 1
        elif net_profit > 0 and operating_cash_flow < net_profit * 0.5:
            risks.append("经营现金流显著低于净利润，利润现金保障倍数偏低")

    # ── Determine level ───────────────────────────────────────
    if data_points < 1:
        solvency_level = "unknown"
        findings.append("财务数据不足，偿债与流动性分析仅供参考")
    elif score >= 3:
        solvency_level = "strong"
    elif score >= 0:
        solvency_level = "moderate"
    else:
        solvency_level = "weak"

    return {
        "solvency_level": solvency_level,
        "liquidity_findings": findings,
        "debt_risks": risks,
    }
