"""
LearnerProfileAnalyzer（学习者画像解析器）

Parses user question and optional profile to determine:
  - knowledge_level: beginner / basic / intermediate / advanced
  - learning_goal: concept explanation / risk awareness / product understanding / ...
  - preferred_style: plain / structured / case-based / terminology
  - missing_context: what we still need to know

Rule-based — no LLM.
"""

from __future__ import annotations

import re


# ── Knowledge level keywords ─────────────────────────────────────

_BEGINNER_PATTERNS = [
    "小白", "完全不懂", "刚入门", "新手", "从零", "零基础",
    "第一次", "什么是", "什么意思", "怎么理解", "不太懂",
    "完全不了解", "没接触过", "不懂", "入门",
]
_BASIC_PATTERNS = [
    "了解一些", "学过一点", "听说过", "大概知道", "基本概念",
    "买过基金", "买过理财", "定投过", "有点了解",
]
_INTERMEDIATE_PATTERNS = [
    "有一定了解", "做过投资", "配置过", "了解基金", "知道风险",
    "看过财报", "会看指标", "年化", "夏普", "回撤",
]
_ADVANCED_PATTERNS = [
    "专业", "从业", "CFA", "基金经理", "量化", "衍生品",
    "对冲", "期权策略", "因子模型", "资产定价",
]

# ── Learning goal keywords ────────────────────────────────────────

_GOAL_PATTERNS: list[tuple[str, str]] = [
    ("入门学习|新手入门|理财入门|投资入门|基金入门|股票入门", "入门学习"),
    ("什么是|什么意思|怎么理解|如何理解|解释|定义|概念", "概念解释"),
    ("防诈骗|防骗|识别骗|骗局|被骗|骗术|诈骗案|金融诈骗", "防诈骗"),
    ("风险|亏损|亏钱|陷阱|注意什么|亏钱案例", "风险识别"),
    ("基金.*怎么|基金.*如何|债券.*怎么|股票.*怎么|保险.*怎么|怎么买|如何买|怎么选", "产品理解"),
    ("理财规划|理财计划|资产配置|怎么分配|怎么安排|财务规划", "理财规划基础"),
    ("赚钱|收益|回报|利息|分红|涨|牛市", "收益认知"),
    ("净值|估值|指数|市盈|市净|ROE|波动", "术语解释"),
]

# ── Preferred style keywords ──────────────────────────────────────

_STYLE_PATTERNS: list[tuple[str, str]] = [
    ("通俗|简单|大白话|容易懂|讲人话|通俗易懂", "通俗解释"),
    ("列出来|有哪些|分几点|步骤|方法|清单", "结构化清单"),
    ("举例|例子|案例|比如|打个比方|实际", "案例化解释"),
    ("术语|专业词|名词解释|什么意思|定义", "术语解释"),
]


def analyze(*, question: str, user_profile: dict | None = None, **kwargs) -> dict:
    """Analyze learner profile from question and optional user_profile.

    Args:
        question: User's consultation question.
        user_profile: Optional dict with knowledge_level, learning_goal, etc.

    Returns:
        dict with knowledge_level, learning_goal, preferred_style, missing_context.
    """
    profile = user_profile or {}
    q = question or ""

    # ── Knowledge level ──────────────────────────────────────────
    knowledge_level = profile.get("knowledge_level", "")
    if not knowledge_level:
        knowledge_level = _detect_knowledge_level(q)

    # ── Learning goal ────────────────────────────────────────────
    learning_goal = profile.get("learning_goal", "")
    if not learning_goal:
        learning_goal = _detect_learning_goal(q)

    # ── Preferred style ──────────────────────────────────────────
    preferred_style = profile.get("preferred_style", "")
    if not preferred_style:
        preferred_style = _detect_style(q)

    # ── Missing context ──────────────────────────────────────────
    missing_context: list[str] = []
    if knowledge_level == "unknown":
        missing_context.append("knowledge_level")
    if not learning_goal or learning_goal == "unknown":
        missing_context.append("learning_goal")

    return {
        "knowledge_level": knowledge_level,
        "learning_goal": learning_goal,
        "preferred_style": preferred_style,
        "missing_context": missing_context,
    }


def _detect_knowledge_level(question: str) -> str:
    """Detect knowledge level from question text."""
    scores: dict[str, int] = {}

    for pat in _ADVANCED_PATTERNS:
        if re.search(pat, question):
            scores["advanced"] = scores.get("advanced", 0) + 1
    for pat in _INTERMEDIATE_PATTERNS:
        if re.search(pat, question):
            scores["intermediate"] = scores.get("intermediate", 0) + 1
    for pat in _BASIC_PATTERNS:
        if re.search(pat, question):
            scores["basic"] = scores.get("basic", 0) + 1
    for pat in _BEGINNER_PATTERNS:
        if re.search(pat, question):
            scores["beginner"] = scores.get("beginner", 0) + 1

    if not scores:
        return "beginner"  # Default to beginner for education intent

    # Return the highest matched level
    for level in ["advanced", "intermediate", "basic", "beginner"]:
        if level in scores:
            return level
    return "beginner"


def _detect_learning_goal(question: str) -> str:
    """Detect primary learning goal from question."""
    for pattern, goal in _GOAL_PATTERNS:
        if re.search(pattern, question):
            return goal
    return "概念解释"  # Default


def _detect_style(question: str) -> str:
    """Detect preferred explanation style from question."""
    for pattern, style in _STYLE_PATTERNS:
        if re.search(pattern, question):
            return style
    return "通俗解释"  # Default
