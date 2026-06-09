"""
Unified Agent Router — deterministic keyword-based intent classifier.

Extends the existing lightweight IntentRouter with:
  - Weighted keyword scoring (same keyword registry as IntentRouter)
  - Priority-based conflict resolution for close scores
  - Normalized confidence scores
  - Candidate list ranking
  - Human-readable reason strings

Design:
  - Purely deterministic; no LLM calls, no external API.
  - Reuses the existing keyword registry from IntentRouter to stay consistent.
  - Adds priority tiers for ties and multi-intent queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.agent_query import RouterCandidate, RouterDecision


# ── Keyword registry (shared with IntentRouter) ──────────────────

_KEYWORD_REGISTRY: dict[str, dict[str, float]] = {
    "advisory": {
        # Core advisory signals — strong
        "投资": 1.0, "理财": 1.0, "资产配置": 2.0, "定投": 1.0,
        "持仓": 1.0, "风险偏好": 1.5, "组合": 0.8, "配置": 0.8,
        "仓位": 1.0, "分散": 0.8,
        "月薪": 0.8, "买房": 0.8, "养老": 0.8, "教育金": 1.0,
        "如何投资": 1.2, "再平衡": 2.0,
        # Generic financial terms — intentionally lower to avoid
        # stealing compliance / education / risk queries.
        "基金": 0.3, "股票": 0.2, "债券": 0.3, "收益": 0.2,
        "买入": 0.3, "卖出": 0.3,
        "长期持有": 1.2, "长期投资": 1.2,
    },
    "financial_report": {
        "财报": 1.5, "年报": 1.5, "季报": 1.5, "招股": 1.0,
        "营收": 1.2, "利润": 1.0, "净利润": 1.2, "ROE": 1.5,
        "负债": 1.0, "资产负债": 1.5, "现金流": 1.2, "毛利": 1.0,
        "杜邦": 1.5, "同比": 1.2, "环比": 1.2, "eps": 1.5,
        "商誉": 1.2, "应收账款": 1.5, "存货": 1.0, "摊销": 1.0,
        "损益表": 1.5, "资产负债表": 1.5, "现金流量表": 1.5,
        "业绩": 0.8, "上市公司": 0.5,
        "EBITDA": 1.5, "周转": 1.0, "毛利率": 1.2,
    },
    "risk_control": {
        "风控": 1.5, "风险": 1.0, "预警": 1.5, "踩雷": 1.0,
        "造假": 1.2, "财务造假": 1.5, "暴雷": 1.2, "违约": 1.0,
        "M-Score": 1.5, "Z-Score": 1.5, "关联交易": 1.5,
        "信用风险": 1.5, "市场风险": 1.5, "操作风险": 1.5,
        "流动性风险": 1.5, "系统性风险": 1.5, "非系统性": 1.0,
        "评级": 0.8, "风险评估": 1.5, "风险审查": 1.5,
        "黑天鹅": 1.5, "灰犀牛": 1.5, "回撤": 1.2,
        "持仓集中度": 2.0, "集中度风险": 2.0, "集中度": 1.2,
        "风险管理": 1.5, "风险控制": 1.5,
        "杠杆": 1.2, "压力测试": 1.5, "仓位风险": 1.5,
        # Colloquial risk signals
        "大跌": 1.2, "太集中": 1.5, "止损": 1.2, "追涨杀跌": 1.2,
        "波动": 0.8, "崩盘": 1.5, "跳水": 1.0,
    },
    "compliance": {
        "合规": 1.5, "法规": 1.5, "条例": 1.0, "监管": 1.5,
        "证监会": 1.5, "央行": 1.5, "银保监": 1.5, "金融监管": 1.5,
        "备案": 1.0, "披露": 0.8, "信息披露": 1.5,
        "证券法": 1.5, "基金法": 1.5, "资管新规": 1.5,
        "反洗钱": 1.5, "数据安全法": 1.5, "个人信息保护": 1.5,
        "私募": 0.8, "登记": 0.5, "牌照": 0.8,
        # User-facing compliance signals
        "推荐": 1.5, "荐股": 2.0, "靠谱": 0.8,
        "适当性": 1.5, "投资者适当性": 2.0, "销售机构": 1.0,
        "投资者保护": 1.5, "保证": 1.5, "年化": 1.0,
        "亏钱": 0.8, "责任": 1.0, "合规红线": 2.0,
        "不能": 0.3, "允许": 0.3,
        # Strong compliance signals
        "营销话术": 2.0, "违规": 2.0, "目标价": 2.0,
        "买入建议": 1.5, "风险揭示": 1.5,
    },
    "education": {
        "什么是": 1.5, "如何": 0.5, "怎么": 0.5, "入门": 1.5,
        "基础": 1.2, "新手": 1.5, "小白": 1.5, "知识": 1.2,
        "科普": 1.5, "学习": 1.2, "解释": 1.0, "区别": 1.0,
        "基金是什么": 1.5, "股票是什么": 1.5, "债券是什么": 1.5,
        "打新": 1.0, "分红": 0.8, "除权": 1.0, "复权": 1.0,
        "市盈率": 1.2, "市净率": 1.2, "ETF": 1.0, "LOF": 1.0,
        "定投": 0.8, "网格": 1.0, "期货": 1.0, "期权": 1.0,
        "保险": 0.8,
        # Finance-specific education signals
        "净值": 1.5, "基金净值": 2.0, "估值": 1.0,
        "怎么看": 0.8, "什么意思": 1.0, "不太懂": 1.2,
        "从哪开始": 1.5, "完全不懂": 1.5,
        # Scam / fraud education
        "防诈骗": 2.0, "骗局": 2.0, "诈骗": 2.0,
        "基金定投原理": 1.5,
    },
}

# ── Intent priority tiers (higher = wins on tie/close score) ─────
# Used as a tiebreaker: priority_adjustment = PRIORITY[intent] * 0.01

INTENT_PRIORITY: dict[str, int] = {
    "compliance": 100,
    "financial_report": 80,
    "risk_control": 70,
    "advisory": 60,
    "education": 50,
}

# ── Intent display names (Chinese) ───────────────────────────────

INTENT_DISPLAY_NAMES: dict[str, str] = {
    "advisory": "智能投顾",
    "financial_report": "财报分析",
    "risk_control": "风控审查",
    "compliance": "监管合规",
    "education": "金融科普",
}


def _tokenize(text: str) -> str:
    return text.lower().strip()


def route_intent(
    question: str,
    user_profile: dict | None = None,
) -> RouterDecision:
    """Route a user question to the best-matching intent.

    Algorithm:
      1. For each intent, sum the weights of all matched keywords.
      2. Apply a priority adjustment (0.01 × intent_priority).
      3. Rank candidates by adjusted score, highest first.
      4. Normalize confidence relative to top score.
      5. Build a human-readable reason from top matched keywords.

    Args:
        question: Raw user question text.
        user_profile: Optional user profile (not currently used but
            reserved for future context-aware routing).

    Returns:
        RouterDecision with selected_intent, confidence, reason, and candidates.
    """
    _ = user_profile  # reserved for future context-aware routing
    normalized = _tokenize(question)
    raw_scores: dict[str, float] = {}
    matched_kw: dict[str, list[str]] = {}

    for intent, keywords in _KEYWORD_REGISTRY.items():
        total = 0.0
        matched: list[str] = []
        for kw, weight in keywords.items():
            if _tokenize(kw) in normalized:
                total += weight
                matched.append(kw)
        raw_scores[intent] = total
        matched_kw[intent] = matched

    # Apply priority adjustment (small bias to break ties)
    adjusted: dict[str, float] = {}
    for intent, score in raw_scores.items():
        priority_bias = INTENT_PRIORITY.get(intent, 0) * 0.01
        adjusted[intent] = score + priority_bias

    # ── Build candidates ranked by adjusted score ─────────────────
    sorted_intents = sorted(adjusted.items(), key=lambda kv: kv[1], reverse=True)

    best_intent = sorted_intents[0][0]
    best_score = raw_scores[best_intent]

    # ── If no keywords matched, default to education ──────────────
    if best_score == 0:
        return RouterDecision(
            selected_intent="education",
            confidence=0.0,
            reason="未匹配到明确的关键词，默认识别为金融科普。请提供更多问题细节以获得更精准的智能体匹配。",
            candidates=[
                RouterCandidate(intent=intent, score=round(raw_scores[intent], 2))
                for intent, _ in sorted_intents
            ],
        )

    # ── Normalize confidence ──────────────────────────────────────
    # Use a practical ceiling of 5.0 for raw scores (a question with
    # 3-4 strong keywords typically reaches this range).
    raw_confidence = min(best_score / 5.0, 1.0)

    # Adjust confidence based on gap to 2nd place
    if len(sorted_intents) >= 2 and raw_scores.get(sorted_intents[1][0], 0) > 0:
        second_score = raw_scores[sorted_intents[1][0]]
        gap_ratio = second_score / max(best_score, 0.01)
        # Narrow gap → lower confidence
        if gap_ratio > 0.8:
            raw_confidence *= 0.75
        elif gap_ratio > 0.5:
            raw_confidence *= 0.85
    else:
        # Sole match → boost slightly
        raw_confidence = min(raw_confidence * 1.1, 1.0)

    confidence = round(min(raw_confidence, 1.0), 2)

    # ── Build reason ──────────────────────────────────────────────
    top_kw = matched_kw.get(best_intent, [])[:5]
    display_name = INTENT_DISPLAY_NAMES.get(best_intent, best_intent)
    if top_kw:
        kw_list = "、".join(top_kw[:3])
        reason = f"用户提到了「{kw_list}」等关键词，路由至{display_name}。"
    else:
        reason = f"根据问题特征路由至{display_name}。"

    return RouterDecision(
        selected_intent=best_intent,
        confidence=confidence,
        reason=reason,
        candidates=[
            RouterCandidate(
                intent=intent,
                score=round(raw_scores[intent], 2),
            )
            for intent, _ in sorted_intents
        ],
    )
