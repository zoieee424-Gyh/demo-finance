"""
FundDcaPlanner（基金定投规划器）

Generates fund DCA (Dollar-Cost Averaging) plan suggestions based on user profile,
investment goals, risk level, and asset allocation.

Rule-based for MVP — no LLM calls.
Outputs fund CATEGORIES only, never specific fund names, codes, or platforms.

Output:
  {
    "frequency": "monthly" | "biweekly" | "weekly",
    "amount_ratio": "...",
    "suitable_categories": [...],
    "review_conditions": [...],
    "pause_conditions": [...]
  }
"""
from __future__ import annotations

from typing import Any


# DCA frequency by risk level and liquidity need
_FREQUENCY_MAP: dict[str, str] = {
    "conservative": "monthly",
    "stable": "monthly",
    "balanced": "biweekly",
    "aggressive": "monthly",
}

# Equity ratio cap by risk level (max % of DCA allocation to equity-like funds)
_EQUITY_CAP: dict[str, float] = {
    "conservative": 20.0,
    "stable": 40.0,
    "balanced": 60.0,
    "aggressive": 80.0,
}

# Suitable fund categories by risk level
_SUITABLE_DCA_CATEGORIES: dict[str, list[str]] = {
    "conservative": [
        "短债/固收类基金",
        "货币市场基金",
        "偏债混合基金",
    ],
    "stable": [
        "宽基指数基金类（如沪深300、中证500对应的指数基金）",
        "短债/固收类基金",
        "偏债混合基金",
    ],
    "balanced": [
        "宽基指数基金类",
        "行业指数基金类（分散配置）",
        "平衡混合基金",
        "债券类基金",
    ],
    "aggressive": [
        "宽基指数基金类",
        "行业指数基金类",
        "偏股混合基金",
        "海外指数基金类",
        "债券类基金（少量配置）",
    ],
}

# Review conditions by risk level and goal horizon
_REVIEW_TEMPLATES: dict[str, list[str]] = {
    "conservative": [
        "每6个月复盘一次定投计划",
        "目标期限临近1年时逐步降低权益类比例",
        "检查实际波动是否超出预期",
    ],
    "stable": [
        "每6个月复盘一次定投计划",
        "市场大幅波动时检查组合偏离度",
        "目标期限临近时降低权益类比例",
    ],
    "balanced": [
        "每季度复盘一次定投组合",
        "根据市场估值水平调整定投金额",
        "目标期限临近2年时开始逐步降低权益类比例",
    ],
    "aggressive": [
        "每季度复盘一次定投组合",
        "关注行业轮动和估值变化",
        "目标期限临近3年时开始降低权益类比例",
    ],
}

_PAUSE_CONDITIONS: list[str] = [
    "收入明显下降或失业",
    "紧急备用金不足（少于3个月生活费）",
    "风险承受能力因人生阶段变化而显著变化",
    "目标发生重大变更（如提前购房）",
    "市场出现极端系统性风险信号",
]


def plan_dca(
    profile: dict[str, Any],
    goals: list[dict[str, Any]],
    risk_level: str,
) -> dict[str, Any]:
    """Generate a fund DCA plan.

    Args:
        profile: Structured profile from ProfileAnalyzer.
        goals: Structured goals from GoalPlanner.
        risk_level: Risk level from RiskAssessor.

    Returns:
        DCA plan dict with frequency, amount_ratio, categories, and conditions.
    """
    # Frequency
    frequency = _FREQUENCY_MAP.get(risk_level, "monthly")

    # Amount ratio — adjust for short-term rigid goals
    has_rigid_short_goal = any(
        g.get("priority") == "capital_preservation"
        and g.get("time_horizon_months", 999) <= 36
        for g in goals
    )
    if has_rigid_short_goal:
        amount_ratio = "月结余的20%-40%（因短期刚性目标建议留足流动性）"
    elif risk_level == "aggressive":
        amount_ratio = "月结余的30%-50%"
    else:
        amount_ratio = "月结余的30%-50%"

    # Suitable categories
    categories = _SUITABLE_DCA_CATEGORIES.get(risk_level, _SUITABLE_DCA_CATEGORIES["stable"])

    # Review conditions
    review = list(_REVIEW_TEMPLATES.get(risk_level, _REVIEW_TEMPLATES["stable"]))

    # Add goal-specific review conditions
    for g in goals:
        goal_type = g.get("goal_type", "")
        horizon = g.get("time_horizon_months")
        if goal_type == "house_purchase" and horizon:
            review.append(f"距离购房目标还有{horizon}个月，建议定期检查资产与首付目标差距")

    # Pause conditions
    pause = list(_PAUSE_CONDITIONS)
    income_level = profile.get("income_level", "medium")
    if income_level == "low":
        pause.insert(0, "月收入低于定投金额的3倍时考虑暂停")

    return {
        "frequency": frequency,
        "amount_ratio": amount_ratio,
        "suitable_categories": categories,
        "review_conditions": review,
        "pause_conditions": pause,
    }
