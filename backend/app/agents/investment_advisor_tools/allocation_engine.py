"""
AllocationEngine（资产配置引擎）

Generates asset class allocation ratios based on risk level and investment goals.
Rule-based templates — outputs asset CLASSES only, never specific products.

Output:
  {
    "allocation": [
      {"asset_class": "现金及货币类", "ratio": 40},
      ...
    ],
    "rationale": "..."
  }
"""
from __future__ import annotations

from typing import Any


# ── Allocation templates by risk level ───────────────────────────
# Each template: list of (asset_class, ratio). Sum of ratios = 100.

_ALLOCATION_TEMPLATES: dict[str, list[tuple[str, float]]] = {
    "conservative": [
        ("现金及货币类", 40.0),
        ("短债/固收类", 40.0),
        ("宽基指数基金类", 20.0),
    ],
    "stable": [
        ("现金及货币类", 20.0),
        ("短债/固收类", 35.0),
        ("宽基指数基金类", 30.0),
        ("平衡混合基金类", 15.0),
    ],
    "balanced": [
        ("现金及货币类", 10.0),
        ("债券类", 25.0),
        ("宽基指数基金类", 35.0),
        ("平衡混合基金类", 20.0),
        ("行业指数基金类", 10.0),
    ],
    "aggressive": [
        ("现金及货币类", 5.0),
        ("债券类", 15.0),
        ("宽基指数基金类", 40.0),
        ("平衡混合基金类", 20.0),
        ("行业指数基金类", 15.0),
        ("海外指数基金类", 5.0),
    ],
}

# ── Rationale templates by risk level ────────────────────────────

_RATIONALE_TEMPLATES: dict[str, str] = {
    "conservative": (
        "用户风险偏好保守，投资目标以本金保护为优先。"
        "配置以高流动性、低波动的现金及固收类资产为主，少量宽基指数敞口用于抵御通胀。"
    ),
    "stable": (
        "用户偏好稳健增长，以固收类资产为底仓，搭配宽基指数和平衡混合基金，"
        "在控制波动的前提下追求适度增值。"
    ),
    "balanced": (
        "用户风险承受能力适中，采用股债均衡配置策略，"
        "兼顾资产增值与下行保护，适合中长期投资。"
    ),
    "aggressive": (
        "用户风险承受能力较高且投资期限较长，以权益类资产为主，"
        "通过分散配置宽基、行业和海外指数基金追求长期资本增值。"
    ),
}


def allocate(risk_level: str, goals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Generate asset allocation based on risk level.

    Args:
        risk_level: Risk level from RiskAssessor (conservative/stable/balanced/aggressive).
        goals: Optional list of goals from GoalPlanner (used for rationale customization).

    Returns:
        Allocation result with asset_class items and rationale.
    """
    template = _ALLOCATION_TEMPLATES.get(risk_level, _ALLOCATION_TEMPLATES["stable"])

    allocation = [
        {"asset_class": ac, "ratio": ratio}
        for ac, ratio in template
    ]

    # Verify sum = 100 (defense against template errors)
    total = sum(item["ratio"] for item in allocation)
    if abs(total - 100.0) > 0.01:
        # Normalize
        for item in allocation:
            item["ratio"] = round(item["ratio"] / total * 100, 1)

    # Build rationale
    rationale = _RATIONALE_TEMPLATES.get(risk_level, _RATIONALE_TEMPLATES["stable"])

    # Customize rationale with goal info
    if goals:
        goal_descs = [g.get("description", "") for g in goals if g.get("description")]
        if goal_descs:
            rationale += f" 本配置考虑了用户的{', '.join(goal_descs)}目标。"

    return {
        "allocation": allocation,
        "rationale": rationale,
    }
