"""
GrowthEfficiencyAnalyzer（成长性与经营效率分析器）

Analyzes growth and operating efficiency based on extracted metrics.
Rule-based — no LLM.
"""

from __future__ import annotations


def analyze_growth(
    *,
    metrics: dict,
    **kwargs,
) -> dict:
    """Analyze growth and efficiency from financial metrics.

    Args:
        metrics: dict from FinancialMetricExtractor.

    Returns:
        dict with growth_level, growth_findings, efficiency_concerns.
    """
    findings: list[str] = []
    concerns: list[str] = []

    revenue_growth = metrics.get("revenue_growth")
    profit_growth = metrics.get("profit_growth")
    accounts_receivable = metrics.get("accounts_receivable")
    inventory = metrics.get("inventory")
    revenue = metrics.get("revenue")

    score = 0
    data_points = 0

    # ── Revenue growth ────────────────────────────────────────
    if revenue_growth is not None:
        data_points += 1
        findings.append(f"营收同比增长{revenue_growth}")
        try:
            val = float(revenue_growth.replace("%", "").replace("+", ""))
            if val >= 20:
                score += 2
            elif val >= 10:
                score += 1
            elif val < 0:
                concerns.append("营收同比下滑，需关注市场份额变化")
                score -= 1
        except ValueError:
            pass

    # ── Profit growth ─────────────────────────────────────────
    if profit_growth is not None:
        data_points += 1
        findings.append(f"净利润同比增长{profit_growth}")
        try:
            val = float(profit_growth.replace("%", "").replace("+", ""))
            if val >= 20:
                score += 2
            elif val >= 10:
                score += 1
            elif val < 0:
                concerns.append("净利润同比下滑，盈利能力可能减弱")
                score -= 1
        except ValueError:
            pass

    # ── Revenue vs profit growth divergence ───────────────────
    if revenue_growth is not None and profit_growth is not None:
        try:
            rv = float(revenue_growth.replace("%", "").replace("+", ""))
            pv = float(profit_growth.replace("%", "").replace("+", ""))
            if rv > 0 and pv < 0:
                concerns.append("营收增长但利润下滑，增收不增利，需关注成本费用端压力")
            if pv > rv + 20:
                findings.append("利润增速显著高于营收增速，经营效率持续改善")
                score += 1
        except ValueError:
            pass

    # ── Accounts receivable / inventory warnings ──────────────
    if accounts_receivable is not None and revenue is not None and revenue > 0:
        data_points += 1
        ar_ratio = accounts_receivable / revenue
        if ar_ratio > 0.50:
            concerns.append(
                f"应收账款占营收比例较高（{ar_ratio:.1%}），"
                "需关注回款速度和坏账风险"
            )
            score -= 1
        elif ar_ratio > 0.30:
            concerns.append(f"应收账款占营收{ar_ratio:.1%}，建议关注账龄结构")

    if inventory is not None and revenue is not None and revenue > 0:
        data_points += 1
        inv_ratio = inventory / revenue
        if inv_ratio > 0.50:
            concerns.append(
                f"存货占营收比例较高（{inv_ratio:.1%}），"
                "存在存货跌价或周转不畅的风险"
            )
            score -= 1

    # ── Determine level ───────────────────────────────────────
    if data_points < 2:
        growth_level = "unknown"
        findings.append("财务数据不足，成长性分析仅供参考")
    elif score >= 3:
        growth_level = "strong"
    elif score >= 0:
        growth_level = "moderate"
    else:
        growth_level = "weak"

    return {
        "growth_level": growth_level,
        "growth_findings": findings,
        "efficiency_concerns": concerns,
    }
