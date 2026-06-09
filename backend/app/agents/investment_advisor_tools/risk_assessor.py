"""
RiskAssessor（风险承受评估器）

Evaluates risk tolerance based on user profile and financial goals.
Rule-based — combines risk preference, time horizon, goal rigidity,
and investment experience into a final risk level.

Output:
  {
    "risk_level": "conservative" | "stable" | "balanced" | "aggressive",
    "max_drawdown_tolerance": "low" | "medium" | "high",
    "suitable_assets": [...],
    "unsuitable_assets": [...]
  }
"""
from __future__ import annotations

from typing import Any


# ── Risk scoring factors ─────────────────────────────────────────
# Each factor contributes a score: lower = more conservative

_RISK_PREFERENCE_SCORE: dict[str, int] = {
    "conservative": 0,
    "stable": 1,
    "balanced": 2,
    "aggressive": 3,
}

_EXPERIENCE_SCORE: dict[str, int] = {
    "beginner": -1,
    "experienced": 1,
}

_PRIORITY_SCORE: dict[str, int] = {
    "capital_preservation": -2,
    "balanced": 0,
    "growth": 2,
}

# Time horizon → score adjustment
# Shorter horizon → more conservative
def _time_horizon_score(months: int | None) -> int:
    if months is None:
        return 0
    if months <= 12:
        return -2
    elif months <= 36:
        return -1
    elif months <= 60:
        return 1
    else:
        return 2


# ── Asset class templates by risk level ──────────────────────────

_SUITABLE_ASSETS: dict[str, list[str]] = {
    "conservative": ["现金及货币类", "短债/固收类", "存款类"],
    "stable": ["现金及货币类", "短债/固收类", "宽基指数基金类", "偏债混合基金"],
    "balanced": ["现金及货币类", "债券类", "宽基指数基金类", "平衡混合基金", "行业指数基金"],
    "aggressive": ["现金及货币类", "宽基指数基金类", "行业指数基金", "偏股混合基金", "海外指数基金"],
}

_UNSUITABLE_ASSETS: dict[str, list[str]] = {
    "conservative": ["高波动权益类", "单一股票", "期货/期权", "私募股权"],
    "stable": ["单一股票", "期货/期权", "私募股权", "高杠杆产品"],
    "balanced": ["单一股票（重仓）", "期货/期权", "高杠杆产品"],
    "aggressive": ["高杠杆产品", "非法集资类产品"],
}

# ── Drawdown tolerance by risk level ─────────────────────────────

_DRAWDOWN_MAP: dict[str, str] = {
    "conservative": "low",
    "stable": "low",
    "balanced": "medium",
    "aggressive": "high",
}


def assess(profile: dict[str, Any], goals: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess risk tolerance level.

    Args:
        profile: Structured profile from ProfileAnalyzer.
        goals: Structured goals from GoalPlanner.

    Returns:
        Risk assessment dict with risk_level, drawdown tolerance, and asset suitability.
    """
    score = 0

    # Factor 1: Risk preference (strongest signal)
    rp = profile.get("risk_preference", "stable")
    score += _RISK_PREFERENCE_SCORE.get(rp, 1)

    # Factor 2: Investment experience
    exp = profile.get("investment_experience", "beginner")
    score += _EXPERIENCE_SCORE.get(exp, -1)

    # Factor 3: Goal priority (most conservative goal drives the result)
    goal_scores = []
    for g in goals:
        priority = g.get("priority", "balanced")
        goal_scores.append(_PRIORITY_SCORE.get(priority, 0))
        # Time horizon factor per goal
        months = g.get("time_horizon_months")
        goal_scores.append(_time_horizon_score(months))

    if goal_scores:
        # Most conservative goal dominates
        score += min(goal_scores)
    else:
        # No goals detected → assume medium-long term
        score += 0

    # Factor 4: Liquidity need
    liquidity = profile.get("liquidity_need", "medium")
    if liquidity == "high":
        score -= 1
    elif liquidity == "low":
        score += 1

    # ── Map score to risk level ───────────────────────────────
    if score <= -1:
        risk_level = "conservative"
    elif score <= 1:
        risk_level = "stable"
    elif score <= 3:
        risk_level = "balanced"
    else:
        risk_level = "aggressive"

    # ── Build result ──────────────────────────────────────────
    return {
        "risk_level": risk_level,
        "max_drawdown_tolerance": _DRAWDOWN_MAP[risk_level],
        "suitable_assets": _SUITABLE_ASSETS[risk_level],
        "unsuitable_assets": _UNSUITABLE_ASSETS[risk_level],
    }
