"""
LearningPathPlanner（学习路径规划器）

Generates a progressive learning path based on the user's knowledge level
and learning goal. Rule-based — no LLM.

The planner creates a sequence of topics from foundational to advanced,
matching the user's current level and desired learning goal.
"""

from __future__ import annotations


# ── Learning path templates ───────────────────────────────────────

_LEARNING_PATHS: dict[str, dict[str, list[dict]]] = {
    "基金入门": {
        "beginner": [
            {"step": 1, "topic": "什么是投资？为什么要投资？", "difficulty": "入门", "prerequisites": []},
            {"step": 2, "topic": "风险与收益的基本关系", "difficulty": "入门", "prerequisites": ["什么是投资"]},
            {"step": 3, "topic": "基金是什么？集合投资原理", "difficulty": "入门", "prerequisites": ["风险收益基本关系"]},
            {"step": 4, "topic": "基金的类型（货币/债券/混合/股票）", "difficulty": "入门", "prerequisites": ["基金基本概念"]},
            {"step": 5, "topic": "基金的费用（管理费/托管费/申赎费）", "difficulty": "入门", "prerequisites": ["基金类型"]},
            {"step": 6, "topic": "基金定投的原理与优势", "difficulty": "基础", "prerequisites": ["基金费用"]},
            {"step": 7, "topic": "如何选择适合自己风险承受能力的基金", "difficulty": "基础", "prerequisites": ["基金定投"]},
        ],
        "basic": [
            {"step": 1, "topic": "基金类型深度（指数/主动/ETF/LOF）", "difficulty": "基础", "prerequisites": []},
            {"step": 2, "topic": "基金评估指标（收益率/波动率/夏普比率/最大回撤）", "difficulty": "基础", "prerequisites": ["基金类型"]},
            {"step": 3, "topic": "资产配置与基金组合构建", "difficulty": "基础", "prerequisites": ["基金评估"]},
            {"step": 4, "topic": "定投策略优化与止盈止损理念", "difficulty": "进阶", "prerequisites": ["基金组合"]},
        ],
    },
    "风险识别": {
        "beginner": [
            {"step": 1, "topic": "什么是投资风险？", "difficulty": "入门", "prerequisites": []},
            {"step": 2, "topic": "市场风险、信用风险、流动性风险的区别", "difficulty": "入门", "prerequisites": ["投资风险概念"]},
            {"step": 3, "topic": "分散投资：不把鸡蛋放一个篮子", "difficulty": "入门", "prerequisites": ["风险类型"]},
            {"step": 4, "topic": "常见的投资亏损原因", "difficulty": "入门", "prerequisites": ["分散投资"]},
            {"step": 5, "topic": "如何评估自己的风险承受能力", "difficulty": "基础", "prerequisites": ["亏损原因"]},
            {"step": 6, "topic": "识别投资诈骗的常见手法", "difficulty": "基础", "prerequisites": ["风险评估"]},
        ],
    },
    "防诈骗": {
        "beginner": [
            {"step": 1, "topic": "投资理财中的常见骗局类型", "difficulty": "入门", "prerequisites": []},
            {"step": 2, "topic": "如何识别保本高收益骗局", "difficulty": "入门", "prerequisites": ["骗局类型"]},
            {"step": 3, "topic": "虚假交易平台与内幕消息骗局", "difficulty": "入门", "prerequisites": ["保本高收益"]},
            {"step": 4, "topic": "老师带单/群里荐股骗局揭秘", "difficulty": "入门", "prerequisites": ["虚假平台"]},
            {"step": 5, "topic": "被骗后的正确应对措施", "difficulty": "基础", "prerequisites": ["荐股骗局"]},
            {"step": 6, "topic": "如何选择正规金融机构和产品", "difficulty": "基础", "prerequisites": ["应对措施"]},
        ],
        "basic": [
            {"step": 1, "topic": "金融诈骗的心理学原理", "difficulty": "基础", "prerequisites": []},
            {"step": 2, "topic": "私募/信托/外汇等领域的常见骗局", "difficulty": "基础", "prerequisites": ["诈骗心理"]},
            {"step": 3, "topic": "非法集资与庞氏骗局的识别", "difficulty": "基础", "prerequisites": ["高级骗局"]},
        ],
    },
    "default": {
        "beginner": [
            {"step": 1, "topic": "什么是投资？基本概念入门", "difficulty": "入门", "prerequisites": []},
            {"step": 2, "topic": "风险与收益：投资的第一课", "difficulty": "入门", "prerequisites": ["投资概念"]},
            {"step": 3, "topic": "常见投资工具概览（存款/基金/债券/股票）", "difficulty": "入门", "prerequisites": ["风险收益"]},
            {"step": 4, "topic": "基金入门：最友好的投资起点", "difficulty": "入门", "prerequisites": ["投资工具"]},
            {"step": 5, "topic": "定投策略：简单有效的长期投资方式", "difficulty": "入门", "prerequisites": ["基金入门"]},
            {"step": 6, "topic": "投资中的常见误区与防骗指南", "difficulty": "基础", "prerequisites": ["定投策略"]},
        ],
        "basic": [
            {"step": 1, "topic": "资产类别深度理解（股票/债券/基金/保险）", "difficulty": "基础", "prerequisites": []},
            {"step": 2, "topic": "资产配置的核心原则", "difficulty": "基础", "prerequisites": ["资产类别"]},
            {"step": 3, "topic": "投资组合风险管理基础", "difficulty": "基础", "prerequisites": ["资产配置"]},
            {"step": 4, "topic": "行为金融：投资中的心理偏差", "difficulty": "进阶", "prerequisites": ["风险管理"]},
        ],
        "intermediate": [
            {"step": 1, "topic": "现代投资组合理论入门", "difficulty": "进阶", "prerequisites": []},
            {"step": 2, "topic": "因子投资与Smart Beta策略", "difficulty": "进阶", "prerequisites": ["组合理论"]},
            {"step": 3, "topic": "行为金融与市场异象", "difficulty": "进阶", "prerequisites": ["因子投资"]},
        ],
    },
}


def plan(
    *,
    question: str,
    learner_profile: dict | None = None,
    concepts: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Generate a personalized learning path.

    Args:
        question: User's question.
        learner_profile: Output from learner_profile_analyzer.
        concepts: Output from concept_explainer.

    Returns:
        dict with learning_steps, estimated_difficulty, next_topics.
    """
    profile = learner_profile or {}
    knowledge_level = profile.get("knowledge_level", "beginner")
    learning_goal = profile.get("learning_goal", "概念解释")

    # Select path template
    path_key = "default"
    if "基金" in learning_goal or "基金" in (question or ""):
        path_key = "基金入门"
    elif "防骗" in learning_goal or "诈骗" in (question or ""):
        path_key = "防诈骗"
    elif "风险" in learning_goal or "风险" in (question or ""):
        path_key = "风险识别"

    template = _LEARNING_PATHS.get(path_key, _LEARNING_PATHS["default"])
    level_steps = template.get(knowledge_level, template.get("beginner", []))

    # Determine estimated difficulty
    difficulty_order = ["入门", "基础", "进阶", "高级"]
    step_difficulties = [s.get("difficulty", "入门") for s in level_steps]
    max_diff = "入门"
    for d in difficulty_order:
        if d in step_difficulties:
            max_diff = d

    estimated_difficulty = max_diff

    # Suggest next topics after this path
    next_topics: list[str] = []
    if path_key == "基金入门":
        next_topics = ["指数基金与ETF", "债券与固定收益基础", "资产配置入门"]
    elif path_key == "防诈骗":
        next_topics = ["投资者保护与维权途径", "正规金融机构识别", "个人财务安全基础"]
    elif path_key == "风险识别":
        next_topics = ["资产配置与风险管理", "投资组合多样化", "行为金融入门"]
    else:
        next_topics = ["基金入门", "风险识别", "资产配置基础"]

    return {
        "learning_steps": level_steps,
        "step_count": len(level_steps),
        "estimated_difficulty": estimated_difficulty,
        "next_topics": next_topics,
        "education_note": "学习路径为教育建议，不构成投资操作指令。请根据自身情况自主安排学习进度。",
    }
