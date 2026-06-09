"""
SecondaryIntentDetector — rule-based keyword detection for secondary intents.

Design:
  - Pure keyword matching, no LLM calls.
  - Input: query string + optional primary intent.
  - Output: deduplicated list of secondary intents (excluding primary).
  - Stable ordering: by keyword match count desc, then lexicographic.

Usage:
    detector = SecondaryIntentDetector()
    secondary = detector.detect("定投基金有风险吗？合规吗？", primary="advisory")
    # → ["risk_control", "compliance"]
"""
from __future__ import annotations


# ── Valid intent labels ──────────────────────────────────────────────

VALID_INTENTS = {"advisory", "financial_report", "risk_control", "compliance", "education"}


# ── Secondary keyword registry ───────────────────────────────────────
#
# Each intent has a list of (keyword, weight) tuples.
# Higher weight = stronger signal that this domain is relevant.
# Keywords are matched case-insensitively as substrings.

_SECONDARY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "risk_control": [
        ("风险", 0.3),
        ("风险大", 0.8),
        ("风险管理", 1.0),
        ("亏损", 0.6),
        ("回撤", 0.8),
        ("波动", 0.5),
        ("大跌", 0.7),
        ("下跌", 0.5),
        ("止损", 0.8),
        ("违约", 0.6),
        ("流动性", 0.8),
        ("集中", 0.5),
        ("太集中", 0.8),
        ("追涨杀跌", 1.0),
        ("黑天鹅", 0.8),
        ("系统性风险", 1.0),
        ("信用风险", 1.0),
        ("市场风险", 1.0),
        ("集中度", 0.8),
        ("暴跌", 0.7),
        ("崩盘", 0.8),
        ("跳水", 0.6),
    ],
    "compliance": [
        ("合规", 0.8),
        ("合规吗", 1.0),
        ("监管", 0.6),
        ("推荐", 0.5),
        ("荐股", 1.0),
        ("个股", 0.6),
        ("保证收益", 1.0),
        ("保本", 0.8),
        ("承诺收益", 1.0),
        ("年化", 0.5),
        ("责任", 0.5),
        ("适当性", 0.8),
        ("销售机构", 0.8),
        ("投资建议靠谱吗", 0.8),
        ("靠谱吗", 0.5),
        ("不允许", 0.5),
        ("能不能买", 0.3),
        ("能买吗", 0.5),
        ("违规", 0.8),
        ("红线", 0.6),
    ],
    "education": [
        ("什么是", 0.8),
        ("是什么", 0.8),
        ("区别", 0.6),
        ("入门", 0.8),
        ("小白", 0.8),
        ("不懂", 0.6),
        ("怎么看", 0.6),
        ("解释", 0.6),
        ("学理财", 0.8),
        ("基础知识", 0.8),
        ("新手", 0.8),
        ("学习", 0.5),
        ("什么意思", 0.6),
        ("不太懂", 0.8),
        ("完全不懂", 0.8),
        ("怎么算", 0.5),
        ("科普", 0.8),
        ("区别是", 0.8),
    ],
    "advisory": [
        ("配置", 0.4),
        ("资产配置", 1.0),
        ("定投", 0.6),
        ("养老", 0.5),
        ("买房", 0.5),
        ("教育金", 0.8),
        ("再平衡", 0.8),
        ("投资组合", 0.8),
        ("理财", 0.4),
        ("长期投资", 0.8),
        ("分散投资", 0.8),
        ("分散化", 0.8),
        ("持仓", 0.4),
        ("组合", 0.3),
        ("资产", 0.3),
        ("配置资产", 0.6),
        ("积累", 0.3),
        ("月薪", 0.5),
        ("工资", 0.4),
        ("储蓄", 0.3),
        ("买入", 0.4),
        ("卖出", 0.4),
        ("止损", 0.5),
        ("止盈", 0.5),
        ("仓位", 0.5),
        ("加仓", 0.5),
        ("减仓", 0.5),
        ("调仓", 0.5),
        ("补仓", 0.4),
        ("持有", 0.3),
    ],
    "financial_report": [
        ("财报", 1.0),
        ("年报", 1.0),
        ("现金流", 0.8),
        ("ROE", 1.0),
        ("利润", 0.5),
        ("资产负债表", 1.0),
        ("利润表", 1.0),
        ("季报", 0.8),
        ("营收", 0.8),
        ("负债", 0.5),
        ("毛利率", 0.8),
        ("杜邦", 0.8),
        ("同比", 0.6),
        ("环比", 0.6),
        ("eps", 0.8),
    ],
}


class SecondaryIntentDetector:
    """Detect secondary (cross-domain) intents from a user query.

    Does NOT call LLM.  Pure keyword matching.
    Secondary intents are returned deduplicated, excluding the
    primary intent (if provided).

    Usage::

        detector = SecondaryIntentDetector()
        secondary = detector.detect(
            "定投基金有风险吗？合规吗？",
            primary="advisory",
        )
        # → ["risk_control", "compliance"]
    """

    def detect(
        self,
        query: str,
        primary_intent: str | None = None,
    ) -> list[str]:
        """Return ordered, deduplicated secondary intents for *query*.

        Args:
            query: Raw user question text.
            primary_intent: The primary intent (from IntentRouter).  This
                            intent is **excluded** from the result.

        Returns:
            List of intent label strings, ordered by keyword match strength
            descending.  The primary intent is never included.
        """
        normalized = query.lower().strip()
        primary = primary_intent or ""

        # Score each intent
        scored: list[tuple[str, float, int]] = []  # (intent, weight_sum, match_count)
        for intent, keywords in _SECONDARY_KEYWORDS.items():
            if intent == primary:
                continue
            weight_sum = 0.0
            match_count = 0
            for kw, weight in keywords:
                if kw.lower() in normalized:
                    weight_sum += weight
                    match_count += 1
            if weight_sum > 0:
                scored.append((intent, weight_sum, match_count))

        # Sort: highest summed weight first; ties broken by match count desc,
        # then lexicographic for stability.
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))

        # Return deduplicated intent labels (already unique by construction)
        return [s[0] for s in scored]
