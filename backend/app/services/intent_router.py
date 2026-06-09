"""
Lightweight keyword-based intent router.

Routes user questions to one of five financial service intents:
  advisory | financial_report | risk_control | compliance | education

Design:
  - Purely keyword-driven; no ML models, no external API calls.
  - Each intent has a curated keyword list with associated weights.
  - Score = sum of weights for matched keywords (case-insensitive).
  - Returns the intent string label with the highest score.
  - This is intentionally simple — no Agent business logic lives here.

Interface contract (expected by ConsultationService):
  IntentRouter().route(question: str) -> str
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Intent labels (matching pre-existing consultation.py Literal) ─

INTENT_LABELS = ["advisory", "financial_report", "risk_control", "compliance", "education"]


# ── Keyword registry ─────────────────────────────────────────────

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
    },
}


def _tokenize(text: str) -> str:
    return text.lower().strip()


class IntentRouter:
    """Lightweight keyword-based intent classifier.

    Usage:
        router = IntentRouter()
        intent: str = router.route("我月薪1万，如何配置资产？")
        # → "advisory"
    """

    def route(self, question: str) -> str:
        """Route a user question to the best-matching intent label.

        Args:
            question: Raw user question text.

        Returns:
            One of: "advisory", "financial_report", "risk_control", "compliance", "education".
        """
        normalized = _tokenize(question)
        scores: dict[str, float] = {}
        matched_kw: dict[str, list[str]] = {}

        for intent, keywords in _KEYWORD_REGISTRY.items():
            total = 0.0
            matched: list[str] = []
            for kw, weight in keywords.items():
                if _tokenize(kw) in normalized:
                    total += weight
                    matched.append(kw)
            scores[intent] = total
            matched_kw[intent] = matched

        best_intent = max(scores, key=lambda i: scores[i])
        best_score = scores[best_intent]

        # If no keywords matched, default to education (safest fallback)
        if best_score == 0:
            return "education"

        return best_intent
