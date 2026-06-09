"""
RiskControlCompliancePolicy（风控合规审查器）

Reviews risk control output for compliance violations.
Rule-based — no LLM.

Detects:
  - Individual stock recommendations
  - Buy/sell/hold/add/reduce position instructions
  - Target prices
  - Definite price predictions
  - Return promises
"""

from __future__ import annotations

import re


def review(*, answer: str, **kwargs) -> dict:
    """Review risk control analysis output for compliance violations.

    Args:
        answer: Full report text to review.

    Returns:
        dict with warnings, risk_notice, is_compliant.
    """
    warnings: list[str] = []

    rules: list[tuple[str, str]] = [
        # Stock codes
        (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "检测到A股股票代码"),
        # Stock recommendations
        (r"推荐买入|建议买入|强烈推荐|强烈建议买入", "检测到个股买入推荐"),
        (r"推荐卖出|建议卖出|建议清仓", "检测到个股卖出建议"),
        # Position instructions
        (r"(建议|推荐)[\s\S]{0,6}(加仓|减仓|满仓|空仓|建仓|平仓)", "检测到仓位操作指令"),
        # Investment ratings
        (r"买入评级|卖出评级|增持评级|减持评级|持有评级", "检测到投资评级表述"),
        # Target price
        (r"目标价\s*\d+", "检测到目标价预测"),
        # Price prediction
        (r"(一定|肯定|势必|必然)(?:涨|跌|上涨|下跌)", "检测到确定性涨跌预测"),
        # Return promise
        (r"保证[收益获利]|稳赚|年化收益\s*\d+%", "检测到收益承诺"),
    ]

    for pattern, desc in rules:
        if re.search(pattern, answer):
            warnings.append(f"[违规] {desc}")

    risk_notice = (
        "投资有风险，入市需谨慎。"
        "本报告仅作风险管理参考，不构成任何投资建议、交易指令或投资决策依据。"
        "风险分析基于有限输入信息，可能未覆盖全部风险因子，"
        "请结合自身情况独立判断，必要时咨询专业风险管理顾问。"
    )

    return {
        "warnings": warnings,
        "risk_notice": risk_notice,
        "is_compliant": len(warnings) == 0,
    }
