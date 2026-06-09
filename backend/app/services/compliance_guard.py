"""
Compliance Guard — financial regulatory compliance validation.

Checks agent outputs against financial compliance rules:
  1. Stock tipping (荐股) — recommending specific stocks.
  2. Price prediction (涨跌预测) — forecasting price movements.
  3. Return promise (收益承诺) — guaranteeing returns.
  4. Investment decision substitution (替用户做投资决策) — making decisions for users.

All investment-related answers receive a mandatory risk notice.
Warnings are collected as strings and added to the response.

Interface contract (expected by ConsultationService):
  ComplianceGuard().review(response: ConsultationResponse) -> ConsultationResponse

Design:
  - Rule-based regex pattern matching. No external API calls.
  - Applied AFTER agent answer generation, BEFORE returning to user.
  - Warnings are added to response.warnings (list[str]).
"""
from __future__ import annotations

import re
from app.schemas.consultation import ConsultationResponse
from app.core.config import RISK_NOTICE


# ── Rule definitions ─────────────────────────────────────────────
# (rule_id, severity, regex_pattern, warning_message)

_RULES: list[tuple[str, str, str, str]] = [
    # ── Stock tipping (荐股) ──────────────────────────────────
    (
        "CMP-001",
        "violation",
        r"推荐\s*(买入|卖出|持有|加仓|减仓)",
        "[违规] 检测到明确的交易操作建议（推荐买入/卖出/持有），本平台不提供个股交易建议。",
    ),
    (
        "CMP-002",
        "violation",
        r"建议\s*(买入|卖出|入手|抛售|建仓)\s*.{0,10}(股票|个股|股份)",
        "[违规] 检测到个股买卖建议，本平台不荐股，请勿将 AI 输出作为交易决策依据。",
    ),
    (
        "CMP-003",
        "warning",
        r"(强烈推荐|强烈建议|一定要买|赶紧|快买|必买|潜力股|牛股)",
        "[风险] 检测到带有强烈倾向的荐股表达，请勿轻信。个股投资需独立判断。",
    ),
    (
        "CMP-004",
        "warning",
        r"代码[是为]?\s*\d{6}",
        "[风险] 检测到具体股票代码，请确认未构成个股推荐。",
    ),

    # ── Price prediction (涨跌预测) ────────────────────────────
    (
        "CMP-010",
        "violation",
        r"(肯定|一定|必然|绝对|100%|百分百).{0,10}(上涨|下跌|涨|跌|翻倍)",
        "[违规] 检测到确定性的涨跌预测，本平台不预测市场走势。",
    ),
    (
        "CMP-011",
        "violation",
        r"(目标价|目标价格)\S*(元|美元|港元)",
        "[违规] 检测到具体目标价位预测，本平台不提供价格预测。",
    ),
    (
        "CMP-012",
        "warning",
        r"(预计|预期|预计).{0,15}(上涨|下跌|涨幅|跌幅)\S*(\d+%|百分之)",
        "[风险] 检测到带有具体数值的涨跌预期，历史走势不代表未来表现。",
    ),
    (
        "CMP-013",
        "warning",
        r"(即将|很快|马上)(大涨|大跌|暴涨|暴跌|反弹|反转)",
        "[风险] 检测到短期走势预判表达，市场短期波动具有高度不确定性。",
    ),

    # ── Return promise (收益承诺) ──────────────────────────────
    (
        "CMP-020",
        "violation",
        r"(保本|保底|兜底|绝对安全|零风险|无风险)",
        "[违规] 检测到保本/兜底承诺，投资产品不存在零风险，禁止承诺保本保收益。",
    ),
    (
        "CMP-021",
        "violation",
        r"(稳赚|躺赚|只赚不赔|稳赢|包赚|旱涝保收)",
        "[违规] 检测到稳赚/包赚等收益承诺表达，所有投资均存在亏损可能。",
    ),
    (
        "CMP-022",
        "violation",
        r"(保证|确保|铁定).{0,10}(收益|回报|盈利|赚钱)",
        "[违规] 检测到收益保证性表述，投资收益具有不确定性，禁止承诺具体回报。",
    ),
    (
        "CMP-023",
        "warning",
        r"(年化|年收益|预期收益|收益率).{0,10}([5-9]\d|\d{3,})\s*%",
        "[风险] 检测到高收益率具体数字，历史收益不代表未来表现，高收益伴随高风险。",
    ),

    # ── Investment decision substitution (替用户决策) ──────────
    (
        "CMP-030",
        "violation",
        r"(你应该|你必须|你一定要|你肯定要|你得).{0,10}(买|卖|投|配置|加仓|减仓|满仓|空仓)",
        "[违规] 检测到替用户做投资决策的表达，AI 不能替代用户做出投资决策。",
    ),
    (
        "CMP-031",
        "violation",
        r"(建议.{0,5}全仓|建议.{0,5}重仓|建议.{0,5}空仓)",
        "[违规] 检测到全仓/重仓/空仓操作建议，此类极端仓位建议风险极高。",
    ),
    (
        "CMP-032",
        "warning",
        r"(赶紧|马上|立刻|赶快).{0,10}(入手|上车|买入|卖出|清仓)",
        "[风险] 检测到催促交易的紧迫性表达，投资决策需冷静判断，不应制造紧迫感。",
    ),
    (
        "CMP-033",
        "warning",
        r"(错过.{0,5}后悔|最后.{0,5}机会|机不可失|时不我待)",
        "[风险] 检测到制造投资焦虑的表达，请保持客观中立的辅助定位。",
    ),

    # ── Generic financial safety ───────────────────────────────
    (
        "CMP-040",
        "warning",
        r"(借钱|贷款|融资|配资).{0,10}(炒股|投资|买基|买股)",
        "[风险] 检测到鼓励杠杆投资（借钱/贷款炒股）的表达，杠杆投资风险极高。",
    ),
    (
        "CMP-041",
        "warning",
        r"(内部消息|内幕|庄家|主力|拉升|出货|跟庄)",
        "[风险] 检测到疑似内幕交易或市场操纵相关用语，相关行为可能涉嫌违法。",
    ),
]


# ── Pre-compiled patterns ────────────────────────────────────────

_COMPILED: list[tuple[str, str, re.Pattern, str]] = []
for rid, sev, pat, msg in _RULES:
    _COMPILED.append((rid, sev, re.compile(pat, re.IGNORECASE), msg))


def _needs_risk_notice(text: str) -> bool:
    """Heuristic: does this text touch on investment topics?"""
    signals = [
        "投资", "理财", "资产", "配置", "基金", "股票", "债券",
        "收益", "风险", "买入", "卖出", "持仓", "仓位", "定投",
        "invest", "portfolio", "allocation",
    ]
    lower = text.lower()
    return any(s.lower() in lower for s in signals)


class ComplianceGuard:
    """Financial compliance guard — reviews agent answers for violations.

    Usage:
        guard = ComplianceGuard()
        checked = guard.review(response)
        # checked.warnings may now contain rule violations
        # checked.risk_notice is populated if needed
    """

    def review(self, response: ConsultationResponse) -> ConsultationResponse:
        """Review an agent response and add warnings + risk notice.

        Args:
            response: The agent's ConsultationResponse to check.

        Returns:
            The same response object, with warnings appended and risk_notice
            set if the answer touches investment topics.
        """
        answer = response.answer
        triggered: set[str] = set()
        new_warnings: list[str] = []

        for rule_id, severity, pattern, message in _COMPILED:
            if pattern.search(answer) and rule_id not in triggered:
                triggered.add(rule_id)
                new_warnings.append(message)

        # Append new warnings to existing ones
        response.warnings = response.warnings + new_warnings

        # Attach risk notice if this is investment-related and not already present
        if _needs_risk_notice(answer) and not response.risk_notice:
            response.risk_notice = RISK_NOTICE

        return response
