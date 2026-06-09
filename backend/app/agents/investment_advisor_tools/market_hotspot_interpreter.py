"""
MarketHotspotInterpreter（市场热点解读器）

Provides neutral interpretation of market hotspots.
Detects hotspot-related keywords in user questions.
Outputs drivers, risk points, and neutral observation suggestions.

Does NOT: predict price movements, recommend individual stocks, or chase hotspots.
"""
from __future__ import annotations

import re
from typing import Any


# Hotspot keyword patterns
_HOTSPOT_PATTERNS: dict[str, re.Pattern] = {
    "ai_tech": re.compile(r"(AI|人工智能|ChatGPT|大模型|GPT|机器学习|深度学习)", re.IGNORECASE),
    "new_energy": re.compile(r"(新能源|光伏|锂电|储能|风电|电动车|新能源汽车|碳中和)"),
    "semiconductor": re.compile(r"(芯片|半导体|集成电路|光刻|晶圆)"),
    "biotech": re.compile(r"(医药|创新药|生物科技|CXO|疫苗|基因)"),
    "consumer": re.compile(r"(消费|白酒|食品|家电|新零售)"),
    "real_estate": re.compile(r"(房地产|楼市|房价|房企|地产)"),
    "finance": re.compile(r"(银行|券商|保险|金融科技|数字货币|区块链)"),
    "sector_rotation": re.compile(r"(板块|行业|赛道|主题|热点|风口|概念)"),
}


# Topic metadata
_TOPIC_INFO: dict[str, dict[str, Any]] = {
    "ai_tech": {
        "topic": "AI与科技板块",
        "drivers": ["政策支持与产业升级预期", "技术突破与应用场景扩展", "全球科技竞争格局"],
        "risk_points": ["高估值回调风险", "短期市场情绪过热", "技术路线不确定性", "个股基本面分化严重"],
        "neutral_view": "AI与科技板块具有长期产业逻辑，但短期估值波动较大。可作为观察主题，并通过分散化资产配置理解相关风险，不建议集中押注单一概念股。",
    },
    "new_energy": {
        "topic": "新能源板块",
        "drivers": ["双碳政策持续推进", "技术降本与规模化", "全球能源转型趋势"],
        "risk_points": ["产能过剩风险", "补贴政策退坡", "技术替代风险", "国际贸易摩擦"],
        "neutral_view": "新能源是长周期产业趋势，但行业已进入分化阶段。适合从产业链、估值和政策变化角度持续观察，避免因单一标的或短期热点集中押注。",
    },
    "semiconductor": {
        "topic": "芯片半导体板块",
        "drivers": ["国产替代政策驱动", "全球供应链重构", "AI算力需求爆发"],
        "risk_points": ["周期性波动明显", "技术壁垒与研发风险", "地缘政治不确定性", "资本开支压力"],
        "neutral_view": "半导体行业具有强周期属性。普通投资者可重点观察产业周期、估值位置和政策变化，避免在情绪高点追涨。",
    },
    "biotech": {
        "topic": "医药生物板块",
        "drivers": ["人口老龄化趋势", "创新药出海逻辑", "政策环境边际改善"],
        "risk_points": ["研发失败风险", "集采政策影响", "估值体系重构", "临床数据不确定性"],
        "neutral_view": "医药板块细分领域差异大，创新药、CXO、医疗器械等各有逻辑。理解主题风险时应关注研发、政策和估值三类因素。",
    },
    "consumer": {
        "topic": "消费板块",
        "drivers": ["消费复苏预期", "居民收入增长", "消费升级/降级分化"],
        "risk_points": ["经济周期敏感性强", "消费信心波动", "行业竞争加剧", "渠道变革冲击"],
        "neutral_view": "消费板块是长期价值投资的重要领域，但需区分必选消费与可选消费的不同逻辑，并关注消费信心、渠道变化和估值波动。",
    },
    "real_estate": {
        "topic": "房地产板块",
        "drivers": ["政策调控边际变化", "城镇化进程", "房企债务化解进度"],
        "risk_points": ["行业基本面尚未企稳", "流动性风险", "政策传导时滞", "人口结构变化"],
        "neutral_view": "房地产行业正处于转型期，短期波动较大，建议保持观望，不因短期政策利好追涨。",
    },
    "finance": {
        "topic": "金融板块",
        "drivers": ["利率环境变化", "资本市场改革", "金融科技应用"],
        "risk_points": ["净息差收窄压力", "资产质量风险", "监管政策变化", "市场情绪波动"],
        "neutral_view": "金融板块与宏观经济高度相关，银行、券商、保险子行业逻辑不同，应分别理解利率、资本市场和资产质量等驱动因素。",
    },
    "sector_rotation": {
        "topic": "板块轮动与主题投资",
        "drivers": ["市场风格切换", "政策催化", "资金流向变化"],
        "risk_points": ["轮动节奏难以把握", "追涨杀跌风险", "短期炒作后回调", "信息不对称"],
        "neutral_view": "板块轮动是市场常态，普通投资者难以精准择时。若关注行业主题，应控制集中度，并以长期资产配置框架约束短期热点冲动。",
    },
}


def interpret(question: str) -> dict[str, Any]:
    """Interpret market hotspots mentioned in the user question.

    Args:
        question: User question text.

    Returns:
        Interpretation dict. If no hotspot detected, is_relevant=False.
    """
    detected_topics: list[str] = []
    for topic_key, pattern in _HOTSPOT_PATTERNS.items():
        if pattern.search(question) and topic_key not in detected_topics:
            detected_topics.append(topic_key)

    if not detected_topics:
        return {
            "is_relevant": False,
            "topic": "",
            "drivers": [],
            "risk_points": [],
            "neutral_view": "本次咨询未涉及当前市场热点主题，建议关注长期资产配置而非短期热点。",
        }

    # Merge info from all detected topics
    merged: dict[str, Any] = {
        "is_relevant": True,
        "topic": "",
        "drivers": [],
        "risk_points": [],
        "neutral_view": "",
    }

    all_drivers: list[str] = []
    all_risks: list[str] = []
    all_views: list[str] = []
    all_topics: list[str] = []
    seen_drivers: set[str] = set()
    seen_risks: set[str] = set()

    for tk in detected_topics[:3]:  # Cap at 3 topics to avoid bloating
        info = _TOPIC_INFO.get(tk)
        if info:
            all_topics.append(info["topic"])
            for d in info["drivers"]:
                if d not in seen_drivers:
                    seen_drivers.add(d)
                    all_drivers.append(d)
            for r in info["risk_points"]:
                if r not in seen_risks:
                    seen_risks.add(r)
                    all_risks.append(r)
            all_views.append(info["neutral_view"])

    merged["topic"] = "、".join(all_topics)
    merged["drivers"] = all_drivers[:6]
    merged["risk_points"] = all_risks[:6]
    merged["neutral_view"] = " ".join(all_views)

    return merged
