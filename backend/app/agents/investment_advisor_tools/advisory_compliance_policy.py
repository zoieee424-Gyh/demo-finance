"""
AdvisoryCompliancePolicy（投顾合规策略器）

Internal compliance gate for the Investment Advisor Agent.
Applied BEFORE the answer is returned — first line of defense.
The global ComplianceGuard runs AFTER as the second line.

Checks:
  - Stock tipping (个股推荐)
  - Price prediction (涨跌预测)
  - Return promise (收益承诺)
  - Investment decision substitution (替用户决策)
  - Missing risk notice (缺少风险提示)

Output:
  {
    "warnings": [...],
    "risk_notice": "...",
    "is_compliant": bool
  }
"""
from __future__ import annotations

import re
from typing import Any


RISK_NOTICE = (
    "投资有风险，入市需谨慎。"
    "本回答仅作信息参考，不构成投资决策依据，"
    "请根据自身风险承受能力独立判断。"
)


# ── Compliance rules ─────────────────────────────────────────────
# (rule_id, severity, regex, warning message)

_RULES: list[tuple[str, str, str, str]] = [
    # Stock tipping
    (
        "ADV-001", "violation",
        r"(推荐|建议).{0,10}(买入|卖出|持有|建仓|加仓|减仓).{0,10}(股票|个股|股份)",
        "[违规] 检测到个股交易建议，投顾不推荐具体股票。请仅提供资产类别配置。",
    ),
    (
        "ADV-002", "violation",
        r"\d{6}\s*[（(].{0,10}[）)]",
        "[违规] 检测到具体股票代码，投顾输出不应包含个股信息。",
    ),
    (
        "ADV-003", "warning",
        r"(牛股|妖股|黑马|白马股|龙头股|概念股).{0,10}(推荐|建议|关注)",
        "[风险] 检测到个股倾向性描述，请避免推荐特定股票。",
    ),

    # Price prediction
    (
        "ADV-010", "violation",
        r"(肯定|一定|必然|必定|绝对).{0,10}(上涨|下跌|涨|跌)",
        "[违规] 检测到确定性涨跌预测，禁止对市场走势做出确定性判断。",
    ),
    (
        "ADV-011", "violation",
        r"目标价.{0,5}\d+",
        "[违规] 检测到具体目标价位，禁止输出价格预测。",
    ),
    (
        "ADV-012", "warning",
        r"(即将|马上|很快).{0,10}(涨|跌|牛|熊)",
        "[风险] 检测到短期走势预判，市场短期波动具有高度不确定性。",
    ),

    # Return promise
    (
        "ADV-020", "violation",
        r"(保本|保底|保证.{0,5}收益|稳赚|包赚|零风险|无风险|绝对安全)",
        "[违规] 检测到收益承诺/保本表述，所有投资均有风险，禁止做出收益保证。",
    ),
    (
        "ADV-021", "warning",
        r"(年化|预期收益|目标收益).{0,10}\d+%",
        '[风险] 检测到具体收益率数字，请补充"历史收益不代表未来表现"提示。',
    ),

    # Decision substitution
    (
        "ADV-030", "violation",
        r"(你应该|你必须|你得|你一定要).{0,10}(买|卖|投|配置|全仓|重仓|空仓)",
        "[违规] 检测到替用户决策的表达，AI 不能替代用户做出投资决策。",
    ),
    (
        "ADV-031", "warning",
        r"(赶紧|赶快|马上|立刻).{0,10}(入手|上车|买入|建仓|加仓)",
        "[风险] 检测到催促交易用语，投资决策应基于理性分析而非紧迫感。",
    ),

    # Risk notice
    (
        "ADV-040", "warning",
        r"^(?!.*(风险提示|风险.*提示|投资有风险|不构成投资建议|仅供参考)).*$",
        # This is a negative lookahead — triggers if NO risk language found
        "[风险] 投资相关输出建议补充风险提示。",
    ),
]


# Pre-compiled patterns
_COMPILED: list[tuple[str, str, re.Pattern, str]] = []
for rid, sev, pat, msg in _RULES:
    try:
        _COMPILED.append((rid, sev, re.compile(pat, re.IGNORECASE), msg))
    except re.error:
        # Skip rules that fail to compile (e.g. complex negative lookaheads)
        pass


def _has_risk_language(text: str) -> bool:
    """Check if the text contains risk disclaimers."""
    risk_phrases = [
        "风险提示", "投资有风险", "不构成投资建议",
        "仅供参考", "入市需谨慎", "风险.*提示",
    ]
    for phrase in risk_phrases:
        if re.search(phrase, text):
            return True
    return False


def review(answer: str) -> dict[str, Any]:
    """Review advisory answer for compliance violations.

    Args:
        answer: The full answer text to check.

    Returns:
        Compliance verdict with warnings, risk_notice, and is_compliant flag.
    """
    warnings: list[str] = []
    triggered: set[str] = set()

    for rule_id, severity, pattern, message in _COMPILED:
        if rule_id in triggered:
            continue
        # Skip the negative-lookahead rule for now, handle separately
        if rule_id == "ADV-040":
            continue
        if pattern.search(answer):
            triggered.add(rule_id)
            warnings.append(message)

    # Special handling: check for missing risk notice
    if not _has_risk_language(answer):
        warnings.append(
            "[风险] 投资相关输出缺少风险提示，请补充'投资有风险，"
            "本回答仅作信息参考，不构成投资决策依据'。"
        )

    # Determine if compliant
    violation_count = sum(1 for w in warnings if w.startswith("[违规]"))
    is_compliant = violation_count == 0

    return {
        "warnings": warnings,
        "risk_notice": RISK_NOTICE,
        "is_compliant": is_compliant,
    }
