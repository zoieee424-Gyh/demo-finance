"""
ProductKnowledgeMapper（金融产品知识映射器）

Maps user-mentioned product categories to knowledge topics.
Explains basic features, main risks, and prerequisite knowledge.

Rule-based — no LLM.
NEVER recommends specific products, codes, or buy/sell timing.
"""

from __future__ import annotations

import re


# ── Product knowledge base ────────────────────────────────────────

_PRODUCT_DB: dict[str, dict] = {
    "货币基金": {
        "category": "货币市场基金",
        "basic_features": [
            "主要投资短期货币市场工具（国债、央行票据、银行存款等）",
            "流动性高，通常T+1/T+0赎回到账",
            "风险较低，但收益也较低",
            "常用于现金管理",
        ],
        "main_risks": [
            "收益率低，长期购买力可能被通胀侵蚀",
            "极端市场条件下净值可能跌破面值（极少见）",
        ],
        "prerequisite_topics": ["什么是基金", "风险与收益关系", "流动性的意义"],
    },
    "债券基金": {
        "category": "债券型基金",
        "basic_features": [
            "主要投资国债、企业债、金融债等固定收益证券",
            "风险和收益介于货币基金和股票基金之间",
            "受利率变化影响（利率上升→债基净值下跌）",
        ],
        "main_risks": [
            "利率风险：利率上升导致债券价格下跌",
            "信用风险：持有的企业债可能违约",
            "流动性风险：部分债券可能难以快速变现",
        ],
        "prerequisite_topics": ["什么是债券", "利率与债券价格关系", "信用评级基础"],
    },
    "指数基金": {
        "category": "指数型基金",
        "basic_features": [
            "跟踪特定指数（如沪深300、中证500），获取市场平均收益",
            "费率低，持仓透明",
            "适合定投和长期持有",
        ],
        "main_risks": [
            "跟随指数波动，指数下跌时基金净值同步下跌",
            "跟踪误差：基金表现与指数存在偏差",
            "不同指数风险差异大（宽基vs行业）",
        ],
        "prerequisite_topics": ["什么是指数", "市场风险与系统性风险", "定投策略基础"],
    },
    "股票基金": {
        "category": "股票型基金",
        "basic_features": [
            "80%以上资产投资于股票",
            "长期预期收益最高，但短期波动最大",
            "分为主动管理型和被动指数型",
        ],
        "main_risks": [
            "市场风险：随股市涨跌大幅波动",
            "风格风险：特定行业或风格的集中风险",
            "管理风险：主动基金依赖基金经理能力",
        ],
        "prerequisite_topics": ["什么是股票", "市场波动与风险承受", "分散投资的重要性"],
    },
    "混合基金": {
        "category": "混合型基金",
        "basic_features": [
            "同时投资股票和债券，比例灵活调整",
            "风险和收益介于股票基金和债券基金之间",
            "分为偏股型、偏债型、平衡型",
        ],
        "main_risks": [
            "股票仓位高时波动较大",
            "资产配置决策依赖基金经理",
            "需关注股债配置比例变化",
        ],
        "prerequisite_topics": ["股票与债券的区别", "资产配置基础", "风险收益关系"],
    },
    "保险": {
        "category": "保险产品",
        "basic_features": [
            "核心功能是风险保障（医疗、意外、身故、财产损失等）",
            "分为消费型（纯保障）和储蓄型（保障+储蓄）",
            "保费支出建议控制在收入5-10%",
        ],
        "main_risks": [
            "储蓄型保险退保损失大",
            "保障范围可能不如预期（注意免责条款）",
            "分红/万能险收益不确定",
        ],
        "prerequisite_topics": ["保险的保障功能", "消费型vs储蓄型", "如何阅读保险条款"],
    },
    "理财产品": {
        "category": "银行理财产品",
        "basic_features": [
            "银行或理财子公司发行的资产管理产品",
            "新规下不再保本保息，净值化管理",
            "风险等级从R1（低风险）到R5（高风险）",
        ],
        "main_risks": [
            "不再保本，净值会波动",
            "部分产品有封闭期，不可提前赎回",
            "收益不固定，可能低于预期",
        ],
        "prerequisite_topics": ["净值化含义", "风险等级R1-R5", "理财vs存款区别"],
    },
}

# ── Keyword to product mapping ────────────────────────────────────

_KEYWORD_MAP: dict[str, str] = {
    "货币基金": "货币基金",
    "货基": "货币基金",
    "余额宝": "货币基金",
    "零钱通": "货币基金",
    "债券基金": "债券基金",
    "债基": "债券基金",
    "纯债": "债券基金",
    "指数基金": "指数基金",
    "指数": "指数基金",
    "沪深300": "指数基金",
    "中证500": "指数基金",
    "ETF": "指数基金",
    "股票基金": "股票基金",
    "股基": "股票基金",
    "主动基金": "股票基金",
    "混合基金": "混合基金",
    "混合": "混合基金",
    "保险": "保险",
    "重疾险": "保险",
    "医疗险": "保险",
    "意外险": "保险",
    "理财产品": "理财产品",
    "银行理财": "理财产品",
    "理财": "理财产品",
}


def map_products(*, question: str, learner_profile: dict | None = None, **kwargs) -> dict:
    """Map product mentions to knowledge categories.

    Args:
        question: User's question.
        learner_profile: Output from learner_profile_analyzer.

    Returns:
        dict with product_categories, basic_features, main_risks, prerequisite_topics.
    """
    q = question or ""
    profile = learner_profile or {}
    knowledge_level = profile.get("knowledge_level", "beginner")

    # Find matching products
    matched_products: list[str] = []
    for keyword, product_name in _KEYWORD_MAP.items():
        if keyword in q:
            if product_name not in matched_products:
                matched_products.append(product_name)

    if not matched_products:
        # Default: provide overview of common products
        matched_products = ["理财产品", "货币基金", "指数基金"]

    product_categories: list[dict] = []
    for p_name in matched_products[:3]:  # Limit to 3
        info = _PRODUCT_DB.get(p_name)
        if info:
            product_categories.append({
                "category": info["category"],
                "basic_features": info["basic_features"],
                "main_risks": info["main_risks"],
                "prerequisite_topics": info["prerequisite_topics"],
            })

    return {
        "product_categories": product_categories,
        "category_count": len(product_categories),
        "education_note": (
            "以上为金融产品类别知识介绍，仅供学习参考。"
            "不构成具体产品推荐、买卖时点建议或投资决策依据。"
            "投资前请根据自身风险承受能力独立判断，必要时咨询专业投资顾问。"
        ),
    }
