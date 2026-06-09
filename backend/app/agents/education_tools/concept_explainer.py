"""
ConceptExplainer (金融概念解释器)

Explains fundamental financial concepts at the user's knowledge level.
Rule-based — no LLM.

Covers: 基金, 债券, 股票, 保险, 指数, 净值, 估值, 风险收益, etc.
"""

from __future__ import annotations

import re


# ── Concept knowledge base ────────────────────────────────────────

_CONCEPT_DB: dict[str, dict] = {
    "基金": {
        "concept": "基金(证券投资基金)",
        "beginner": (
            "基金就是把很多人的钱集合起来，交给专业的基金公司去投资。"
            "你买基金相当于买了一篮子股票或债券，不需要自己研究每一只。"
        ),
        "basic": (
            "基金是集合投资工具，分为公募和私募。按投向分为股票型、债券型、"
            "混合型、货币型。核心优势是分散风险和专业化管理。"
        ),
        "intermediate": (
            "基金通过投资组合分散非系统性风险，管理费、托管费影响净收益。"
            "需关注：基金类型、基金经理、历史业绩、夏普比率、最大回撤、规模。"
        ),
        "advanced": (
            "基金评估需综合：Alpha、Beta、Sharpe、Sortino、最大回撤、"
            "跟踪误差(指数基金)、信息比率。注意幸存者偏差和风格漂移。"
        ),
        "key_points": [
            "基金不等于存款，不保本",
            "净值会波动，短期可能亏损",
            "不同类型基金风险差异大",
            "费用(管理费/托管费/申购赎回费)影响长期收益",
        ],
        "common_misunderstandings": [
            "误区：基金一定赚钱 -> 事实：基金净值随市场波动，可能亏损",
            "误区：净值低的基金更便宜 -> 事实：净值高低与贵贱无关",
            "误区：过去涨得好未来一定好 -> 事实：过往业绩不代表未来表现",
        ],
    },
    "股票": {
        "concept": "股票",
        "beginner": (
            "买股票就是成为一家上市公司的股东，公司赚钱你分红，"
            "股价涨了你能卖出获利。但股价也可能下跌，你投入的本金会亏损。"
        ),
        "basic": (
            "股票代表公司所有权。分为A股、港股、美股等。"
            "核心是公司基本面：盈利能力、成长性、估值水平。"
            "股票投资需要评估风险承受能力。"
        ),
        "intermediate": (
            "股票分析分为基本面分析(财报、行业、竞争力)和技术分析(量价、趋势、指标)。"
            "估值方法包括PE、PB、PEG、DCF等。需关注市场风险、行业风险、个股风险。"
        ),
        "advanced": (
            "股票定价涉及CAPM、多因子模型、行为金融。"
            "需评估：Beta、波动率、流动性、信息不对称。"
            "机构关注ROE、FCF、护城河、管理层质量。"
        ),
        "key_points": [
            "股票价格波动大，可能大幅亏损",
            "个股风险远高于指数基金",
            "不要用生活必需资金投资股票",
            "分散投资降低单一个股风险",
        ],
        "common_misunderstandings": [
            "误区：股票就是赌博 -> 事实：长期持有优质公司股票是理性投资，但需研究基本面",
            "误区：听消息就能赚钱 -> 事实：内幕交易违法，公开消息已反映在价格中",
        ],
    },
    "债券": {
        "concept": "债券",
        "beginner": (
            "债券就是借钱给政府或企业，他们到期还本付息。"
            "债券收益比存款高但比股票低，风险也介于存款和股票之间。"
        ),
        "basic": (
            "债券是固定收益证券，分为国债、地方政府债、企业债、金融债等。"
            "核心关注：发行人信用、利率水平、到期时间。"
            "利率上升时债券价格下跌。"
        ),
        "intermediate": (
            "债券定价基于到期收益率(YTM)、久期和凸性。"
            "信用债需评估信用利差和违约风险。利率敏感性与久期正相关。"
        ),
        "advanced": (
            "债券组合管理涉及久期匹配、收益率曲线策略、信用分析。"
            "高收益债(垃圾债)信用风险显著高于投资级。"
        ),
        "key_points": [
            "债券不是无风险资产(仅国债近似无信用风险)",
            "利率上升->债券价格下跌",
            "信用债存在违约风险",
            "债券收益通常低于股票长期收益",
        ],
        "common_misunderstandings": [
            "误区：债券一定保本 -> 事实：企业债可能违约，利率变化导致价格波动",
            "误区：债券收益固定 -> 事实：价格会波动，实际收益可能偏离票面利率",
        ],
    },
    "保险": {
        "concept": "保险",
        "beginner": (
            "保险是花小钱防大风险。你每年交保费，万一出事了(生病、意外、财产损失)，"
            "保险公司赔你一笔钱。保险的核心功能是保障，不是投资。"
        ),
        "basic": (
            "保险分为人身保险(寿险、健康险、意外险)和财产保险。"
            "核心原则：先保障后理财，保费支出应控制在收入5-10%。"
            "注意区分消费型和储蓄型。"
        ),
        "intermediate": (
            "保险产品设计基于大数法则和精算定价。"
            "需评估：保额充足性、保障范围、免责条款、等待期、现金价值(储蓄型)。"
        ),
        "advanced": (
            "保险组合优化涉及：风险自留额度、免赔额选择、定期vs终身、通胀调整。"
            "保险金信托等高阶工具需专业规划。"
        ),
        "key_points": [
            "保险的核心是保障，不是投资",
            "先保障(医疗/重疾/意外)后理财(年金/分红)",
            "仔细阅读免责条款和等待期",
            "不推荐具体保险产品",
        ],
        "common_misunderstandings": [
            "误区：买了保险就万事大吉 -> 事实：需关注保障范围、保额是否充足",
            "误区：返还型保险一定划算 -> 事实：需计算实际收益率，可能不如消费型+自行投资",
        ],
    },
    "指数": {
        "concept": "指数(股票指数/债券指数)",
        "beginner": (
            "指数就像股市的温度计。沪深300指数反映A股最大的300家公司，"
            "涨了说明大部分大公司股价在涨。指数基金跟踪指数，买入相当于买了整个市场。"
        ),
        "basic": (
            "常见指数：沪深300(大盘蓝筹)、中证500(中盘成长)、创业板指、"
            "科创50、标普500、纳斯达克100。指数基金费率低、分散风险、适合定投。"
        ),
        "intermediate": (
            "指数编制涉及：样本空间、选样方法、加权方式(市值加权/等权/因子加权)、"
            "定期调整。指数投资核心是获取市场Beta收益。"
        ),
        "advanced": (
            "Smart Beta指数基于因子(价值、动量、质量、低波、规模)构建，"
            "介于被动和主动之间。需评估因子暴露、换手率和容量。"
        ),
        "key_points": [
            "指数基金是分散投资的低成本工具",
            "宽基指数比行业指数更分散",
            "指数不能保证赚钱，跟随市场波动",
            "不同指数风险特征不同",
        ],
        "common_misunderstandings": [
            "误区：指数不会跌 -> 事实：指数随市场波动，可能大幅回撤",
            "误区：所有指数基金都一样 -> 事实：跟踪误差、费率、规模差异大",
        ],
    },
    "风险收益": {
        "concept": "风险与收益的关系",
        "beginner": (
            "投资的基本规律：想赚更多钱，就要承担更大的亏损风险。"
            "存款最安全但利息最低，股票可能涨很多也可能亏很多。"
        ),
        "basic": (
            "风险收益呈正相关：现金<货币基金<债券<混合基金<股票<衍生品。"
            "投资前先评估自己最多能承受多少亏损(最大回撤容忍度)。"
        ),
        "intermediate": (
            "风险度量：标准差(波动率)、Beta(系统性风险)、VaR(在险价值)、"
            "最大回撤。收益需风险调整后评估：夏普比率、Sortino比率。"
        ),
        "advanced": (
            "现代投资组合理论(MPT)：通过低相关性资产组合降低非系统性风险。"
            "有效前沿表示给定风险下的最优收益。需考虑肥尾风险和黑天鹅事件。"
        ),
        "key_points": [
            "高收益伴随高风险，没有例外",
            "分散投资降低非系统性风险",
            "不要只看收益不看最大回撤",
            "历史表现不代表未来",
        ],
        "common_misunderstandings": [
            "误区：有风险就是坏事 -> 事实：风险是收益的来源，关键是管理而非消除",
            "误区：分散投资就无风险 -> 事实：系统性风险无法分散",
        ],
    },
}

# ── Fallback concept ──────────────────────────────────────────────

_FALLBACK_CONCEPT = {
    "concept": "金融基础知识",
    "beginner": (
        "这个问题涉及金融基础知识。建议先从什么是投资、"
        "风险与收益的关系、分散投资的重要性这几个基础概念开始学习。"
    ),
    "basic": (
        "该主题建议从基本概念入手，了解核心原理后再深入学习。"
        "可参考投资者教育平台和监管机构发布的投教材料。"
    ),
    "intermediate": (
        "该主题涉及较专业的金融知识，建议结合具体案例和数据加深理解。"
    ),
    "advanced": (
        "该主题需要专业知识背景，建议参考学术文献和专业研究。"
    ),
    "key_points": [
        "学习金融知识应循序渐进",
        "关注监管机构发布的投资者教育材料",
    ],
    "common_misunderstandings": [],
}


def explain(*, question: str, learner_profile: dict | None = None, **kwargs) -> dict:
    """Explain financial concepts detected in the question.

    Args:
        question: User's question.
        learner_profile: Output from learner_profile_analyzer.

    Returns:
        dict with concepts, explanations, key_points, and common_misunderstandings.
    """
    profile = learner_profile or {}
    knowledge_level = profile.get("knowledge_level", "beginner")
    q = question or ""

    # Detect which concepts the question is about
    detected_concepts: list[str] = []
    for keyword in _CONCEPT_DB:
        if keyword in q:
            detected_concepts.append(keyword)

    if not detected_concepts:
        # Try broader matching
        if any(w in q for w in ["投资", "理财", "赚钱", "资产", "配置"]):
            detected_concepts = ["风险收益", "基金"]
        elif any(w in q for w in ["亏", "跌", "损失"]):
            detected_concepts = ["风险收益"]
        else:
            detected_concepts = ["风险收益"]  # Default

    concepts: list[dict] = []
    for c_name in detected_concepts:
        info = _CONCEPT_DB.get(c_name, _FALLBACK_CONCEPT)
        level_text = info.get(knowledge_level, info.get("beginner", ""))
        if not level_text:
            level_text = info.get("beginner", "")

        concepts.append({
            "concept": info["concept"],
            "explanation": level_text,
            "key_points": info.get("key_points", []),
            "common_misunderstandings": info.get("common_misunderstandings", []),
        })

    return {
        "concepts": concepts,
        "concept_count": len(concepts),
        "knowledge_level_used": knowledge_level,
    }
