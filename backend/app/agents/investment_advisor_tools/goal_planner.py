"""
GoalPlanner（投资目标规划器）

Decomposes user financial goals from question text into structured goal objects.
Rule-based for MVP — no LLM calls.

Output: list of Goal dicts:
  {
    "goal_type": "house_purchase" | "retirement" | "education_fund" | "emergency_fund" | "wealth_growth",
    "description": "...",
    "time_horizon_months": int,
    "priority": "capital_preservation" | "balanced" | "growth"
  }
"""
from __future__ import annotations

import re
from typing import Any


# ── Goal type detection ──────────────────────────────────────────

_GOAL_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("house_purchase", "购房", re.compile(r"(买房|购房|首付|房产|住宅)")),
    ("retirement", "养老", re.compile(r"(养老|退休|晚年|养老金)")),
    ("education_fund", "教育金", re.compile(r"(教育|上学|孩子.{0,5}(学|教育)|留学|培训)")),
    ("emergency_fund", "短期备用金", re.compile(r"(应急|备用金|急用|不时之需|短期.{0,3}用)")),
    ("wealth_growth", "财富增值", re.compile(r"(增值|赚钱|收益|翻倍|财富|积累|攒钱)")),
]

# ── Time horizon extraction ──────────────────────────────────────

_TIME_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"(\d+)\s*个?\s*月"), lambda m: int(m.group(1))),
    (re.compile(r"(\d+)\s*年"), lambda m: int(m.group(1)) * 12),
    (re.compile(r"短期"), lambda m: 12),
    (re.compile(r"中长期"), lambda m: 60),
    (re.compile(r"中期"), lambda m: 36),
    (re.compile(r"长期"), lambda m: 120),
]


def _extract_time_horizon(text: str) -> int | None:
    """Extract time horizon in months from text. Returns None if not found."""
    best: int | None = None
    for pattern, extractor in _TIME_PATTERNS:
        m = pattern.search(text)
        if m:
            months = extractor(m)
            # Prefer explicit numeric values over fuzzy terms
            if best is None or (m.group(0) not in ("短期", "中期", "中长期", "长期")):
                best = months
    return best


def _determine_priority(goal_type: str, time_horizon_months: int | None) -> str:
    """Determine investment priority based on goal type and time horizon."""
    # Rigid, near-term goals → capital preservation
    rigid_goals = {"house_purchase", "emergency_fund", "education_fund"}

    if goal_type in rigid_goals:
        if time_horizon_months is not None and time_horizon_months <= 36:
            return "capital_preservation"
        return "balanced"

    if goal_type == "retirement":
        if time_horizon_months is not None and time_horizon_months <= 60:
            return "balanced"
        return "growth"

    # wealth_growth
    if time_horizon_months is not None and time_horizon_months <= 12:
        return "balanced"
    return "growth"


def plan(question: str, profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Extract and structure financial goals from the user question.

    Args:
        question: Raw user question.
        profile: Optional structured profile from ProfileAnalyzer.

    Returns:
        List of goal dicts (may be empty if no clear goal detected).
    """
    goals: list[dict[str, Any]] = []
    detected_types: set[str] = set()

    # ── Detect goal types ────────────────────────────────────
    for goal_type, desc, pattern in _GOAL_PATTERNS:
        m = pattern.search(question)
        if m and goal_type not in detected_types:
            detected_types.add(goal_type)
            time_horizon = _extract_time_horizon(question)

            # Use profile constraints to refine time horizon
            if time_horizon is None and profile:
                constraints = profile.get("constraints", [])
                for c in constraints:
                    t = _extract_time_horizon(str(c))
                    if t is not None:
                        time_horizon = t
                        break

            priority = _determine_priority(goal_type, time_horizon)

            goals.append({
                "goal_type": goal_type,
                "description": desc,
                "time_horizon_months": time_horizon,
                "priority": priority,
            })

    # ── If no goal detected, infer from constraints ──────────
    if not goals and profile:
        constraints = profile.get("constraints", [])
        for c in constraints:
            for goal_type, desc, pattern in _GOAL_PATTERNS:
                if pattern.search(str(c)) and goal_type not in detected_types:
                    detected_types.add(goal_type)
                    t = _extract_time_horizon(str(c))
                    goals.append({
                        "goal_type": goal_type,
                        "description": desc,
                        "time_horizon_months": t,
                        "priority": _determine_priority(goal_type, t),
                    })
                    break

    return goals
