"""
ProfitabilityAnalyzer（盈利能力分析器）

Analyzes profitability based on extracted metrics.
Rule-based — no LLM.

Output:
  {
    "profitability_level": "strong" | "moderate" | "weak" | "unknown",
    "key_findings": list[str],
    "concerns": list[str]
  }
"""

from __future__ import annotations


def analyze_profitability(
    *,
    metrics: dict,
    **kwargs,
) -> dict:
    """Analyze profitability from financial metrics.

    Args:
        metrics: dict from FinancialMetricExtractor.

    Returns:
        dict with profitability_level, key_findings, concerns.
    """
    findings: list[str] = []
    concerns: list[str] = []

    revenue = metrics.get("revenue")
    net_profit = metrics.get("net_profit")
    gross_margin = metrics.get("gross_margin")
    net_margin = metrics.get("net_margin")
    profit_growth = metrics.get("profit_growth")
    revenue_growth = metrics.get("revenue_growth")

    score = 0
    data_points = 0

    # ── Net profit margin ─────────────────────────────────────
    if net_margin is not None:
        data_points += 1
        if net_margin >= 0.20:
            findings.append(f"净利率{net_margin:.1%}，处于较高水平")
            score += 2
        elif net_margin >= 0.10:
            findings.append(f"净利率{net_margin:.1%}，处于合理水平")
            score += 1
        elif net_margin >= 0:
            findings.append(f"净利率{net_margin:.1%}，盈利能力偏弱")
            concerns.append(f"净利率仅{net_margin:.1%}，盈利空间较窄")
            score -= 1
        else:
            findings.append(f"净利率{net_margin:.1%}，处于亏损状态")
            concerns.append("当前净利率为负，企业盈利能力不足")
            score -= 2

    # ── Gross margin ──────────────────────────────────────────
    if gross_margin is not None:
        data_points += 1
        if gross_margin >= 0.40:
            findings.append(f"毛利率{gross_margin:.1%}，产品竞争力强")
            score += 1
        elif gross_margin >= 0.20:
            findings.append(f"毛利率{gross_margin:.1%}，处于行业中等水平")
        else:
            findings.append(f"毛利率{gross_margin:.1%}，产品溢价能力偏弱")
            concerns.append("毛利率偏低，需关注成本控制与定价能力")

    # ── Revenue / profit growth ───────────────────────────────
    if profit_growth is not None:
        data_points += 1
        findings.append(f"净利润同比增长{profit_growth}")
        if profit_growth.startswith("+"):
            score += 1
        elif profit_growth.startswith("-"):
            concerns.append(f"净利润同比下降{profit_growth}")
            score -= 1

    if revenue_growth is not None:
        data_points += 1
        findings.append(f"营收同比增长{revenue_growth}")
        if revenue_growth.startswith("+"):
            score += 1
        elif revenue_growth.startswith("-"):
            concerns.append(f"营收同比下降{revenue_growth}")
            score -= 1

    # ── Revenue vs profit ─────────────────────────────────────
    if revenue is not None and net_profit is not None and revenue > 0:
        data_points += 1
        ratio = net_profit / revenue
        if ratio >= 0.15:
            findings.append("营收利润率较高，成本费用控制良好")
            score += 1
        elif ratio >= 0.05:
            findings.append("营收利润率处于合理区间")
        else:
            concerns.append(f"营收利润率仅{ratio:.1%}，费用端可能存在压力")

    # ── Determine level ───────────────────────────────────────
    if data_points < 2:
        profitability_level = "unknown"
        findings.append("财务数据不足，盈利能力分析仅供参考")
    elif score >= 3:
        profitability_level = "strong"
    elif score >= 0:
        profitability_level = "moderate"
    else:
        profitability_level = "weak"

    return {
        "profitability_level": profitability_level,
        "key_findings": findings,
        "concerns": concerns,
    }
