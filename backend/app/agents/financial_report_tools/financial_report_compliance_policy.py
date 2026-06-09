"""
FinancialReportCompliancePolicy（财报分析合规审查器）

Reviews financial report analysis output for compliance violations.
Rule-based — no LLM.

Detects:
  - Individual stock recommendations
  - Buy/sell/hold ratings
  - Target price predictions
  - Definite price movement predictions
  - Return promises
"""

from __future__ import annotations

import re


def review(
    *,
    answer: str,
    **kwargs,
) -> dict:
    """Review a financial report analysis for compliance violations.

    Args:
        answer: Full report text to review.

    Returns:
        dict with warnings, risk_notice, is_compliant.
    """
    warnings: list[str] = []

    # ── Detection rules ───────────────────────────────────────
    rules: list[tuple[str, str]] = [
        # Stock recommendation
        (r"推荐买入|建议买入|强烈推荐|强烈建议买入", "检测到个股买入推荐"),
        (r"推荐卖出|建议卖出|建议清仓", "检测到个股卖出建议"),
        (r"买入评级|卖出评级|增持评级|减持评级|持有评级", "检测到投资评级表述"),
        # Target price
        (r"目标价\s*\d+|看[涨多]到\s*\d+|上涨空间\s*\d+%", "检测到目标价/上涨空间预测"),
        # Price prediction
        (r"(股价|股票).*(?:一定会|肯定会|势必|必定|必然)(?:涨|跌|上涨|下跌)", "检测到确定性涨跌预测"),
        # Return promise
        (r"保证[年收益获利]|稳赚|年化收益[率]?\s*\d+%", "检测到收益承诺/保证回报表述"),
        # Agent making decisions for user
        (r"(建议|推荐)(您|你)(立即|马上|现在就)(买入|卖出|加仓|减仓)", "检测到替用户决策表述"),
        # Specific stock codes (no word boundary needed for CN text)
        (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "检测到A股股票代码"),
    ]

    for pattern, desc in rules:
        if re.search(pattern, answer):
            warnings.append(f"[违规] {desc}")

    # ── Risk notice ───────────────────────────────────────────
    risk_notice = (
        "投资有风险，入市需谨慎。"
        "本报告仅作财务信息分析参考，不构成任何投资建议、"
        "交易指令或投资评级。"
        "财报数据具有时效性，具体投资决策请结合最新信息独立判断。"
    )

    return {
        "warnings": warnings,
        "risk_notice": risk_notice,
        "is_compliant": len(warnings) == 0,
    }
