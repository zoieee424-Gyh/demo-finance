"""
EducationCompliancePolicy（投教合规审查器）

Reviews education output for boundary violations:
  - Stock/fund code recommendations
  - Return promises
  - Price predictions
  - Replacing user investment decisions
  - Treating education content as investment advice

Rule-based — no LLM.

IMPORTANT: This validator distinguishes between:
  A. Educational content that REFERENCES or EXPLAINS violations as examples
  B. The model ITSELF outputting violation suggestions or promises

Scam education content naturally references scam language as counter-examples;
this is NOT a compliance violation.
"""

from __future__ import annotations

import re


def review(*, answer: str, **kwargs) -> dict:
    """Review education output for compliance violations.

    Args:
        answer: Full education report text.

    Returns:
        dict with warnings, risk_notice, is_compliant.
    """
    warnings: list[str] = []

    # ── Rules: detect model-initiated violations ──────────────────

    rules: list[tuple[str, str]] = [
        # Stock codes (model recommending specific stocks)
        (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "检测到具体股票代码，投教输出不应包含具体代码"),

        # Stock purchase recommendations by the model
        (r"推荐(?:你|您|大家)?(?:买入|购买|买|持有)", "检测到买入/持有推荐，投教不应做投资建议"),
        (r"建议(?:你|您|大家)?(?:买入|卖出|加仓|减仓|满仓|清仓)", "检测到交易操作建议"),

        # Return promises
        (r"保证[收益获利]|保证.*年化|稳赚|包赚|必赚|绝不亏损|只赚不赔", "检测到收益承诺"),

        # Definite price predictions
        (r"(?:一定|肯定|势必|必然|绝对).{0,2}(?:涨|跌|上涨|下跌)", "检测到确定性涨跌预测"),

        # Agent making decisions for user
        (r"(?:我建议你|你最好|你应该|你必须|你一定要)(?:买入|卖出|投资|配置)", "检测到替用户决策表述"),

        # Specific fund code
        (r"基金代码\s*\d{6}", "检测到具体基金代码推荐"),
    ]

    for pattern, desc in rules:
        if re.search(pattern, answer):
            warnings.append(f"[合规警告] {desc}")

    # ── Risk notice ───────────────────────────────────────────────

    risk_notice = (
        "本内容仅供金融知识学习和投资者教育参考，不构成任何投资建议、"
        "产品推荐或投资决策依据。投资有风险，入市需谨慎。"
        "请根据自身风险承受能力独立判断，必要时咨询专业投资顾问。"
    )

    return {
        "warnings": warnings,
        "risk_notice": risk_notice,
        "is_compliant": len(warnings) == 0,
    }
