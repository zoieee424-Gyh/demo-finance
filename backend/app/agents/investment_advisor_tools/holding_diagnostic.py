"""
HoldingDiagnostic（持仓诊断器）

Diagnoses issues in a user''s existing portfolio holdings.
Detects: concentration risk, equity overweight, insufficient liquidity, sum mismatches.

Input: list of holding dicts from request.user_profile["holdings"]
Output: issues, adjustment_directions, risk_flags

Does NOT output buy/sell instructions.
"""
from __future__ import annotations

from typing import Any


# Asset class → category mapping for risk classification
_EQUITY_CLASSES = {
    "宽基指数基金类", "行业指数基金类", "偏股混合基金",
    "海外指数基金类", "平衡混合基金类", "板块/主题ETF类",
}

_FIXED_INCOME_CLASSES = {
    "短债/固收类", "债券类", "偏债混合基金", "存款类",
}

_LIQUID_CLASSES = {
    "现金及货币类", "货币市场基金",
}

# Concentration threshold: single class > this % = concentration risk
_CONCENTRATION_THRESHOLD = 50.0

# Liquidity minimum: at least this % should be liquid
_MIN_LIQUIDITY_RATIO = 5.0


def diagnose(
    holdings: list[dict[str, Any]],
    risk_level: str = "stable",
) -> dict[str, Any]:
    """Diagnose portfolio holdings for issues.

    Args:
        holdings: List of {"asset_class": str, "ratio": float} dicts.
        risk_level: Risk level from RiskAssessor for equity cap lookup.

    Returns:
        Diagnosis dict with issues, adjustment_directions, and risk_flags.
    """
    issues: list[str] = []
    directions: list[str] = []
    flags: list[str] = []

    if not holdings:
        return {
            "issues": [],
            "adjustment_directions": [],
            "risk_flags": [],
        }

    # Check sum
    total = sum(h.get("ratio", 0) for h in holdings)
    if abs(total - 100.0) > 1.0:
        issues.append(f"持仓比例总和为{total:.1f}%，不等于100%，请核实数据完整性。")

    # Compute category totals
    equity_total = sum(h.get("ratio", 0) for h in holdings if h.get("asset_class", "") in _EQUITY_CLASSES)
    fixed_income_total = sum(h.get("ratio", 0) for h in holdings if h.get("asset_class", "") in _FIXED_INCOME_CLASSES)
    liquid_total = sum(h.get("ratio", 0) for h in holdings if h.get("asset_class", "") in _LIQUID_CLASSES)

    # Equity cap by risk level
    _EQUITY_MAX: dict[str, float] = {
        "conservative": 20.0,
        "stable": 45.0,
        "balanced": 65.0,
        "aggressive": 85.0,
    }
    equity_max = _EQUITY_MAX.get(risk_level, 45.0)

    if equity_total > equity_max:
        issues.append(f"权益类资产占比{equity_total:.0f}%，超出{risk_level}型上限{equity_max:.0f}%。")
        directions.append(f"可考虑将权益类资产比例逐步调整至{equity_max:.0f}%以内，并复核固收或现金类资产是否满足目标期限需求。")
        flags.append("equity_overweight")

    # Concentration check
    for h in holdings:
        ratio = h.get("ratio", 0)
        asset_class = h.get("asset_class", "")
        if ratio > _CONCENTRATION_THRESHOLD and asset_class not in _LIQUID_CLASSES:
            issues.append(f"{asset_class}占比{ratio:.0f}%，单一类别集中度过高（>{_CONCENTRATION_THRESHOLD:.0f}%）。")
            directions.append(f"可考虑降低{asset_class}集中度，并通过相关性较低的资产类别分散组合风险。")
            flags.append("concentration_risk")
            break  # One flag is enough

    # Liquidity check
    if liquid_total < _MIN_LIQUIDITY_RATIO:
        issues.append(f"现金及货币类资产仅占比{liquid_total:.0f}%，流动性储备不足。")
        directions.append("可考虑保留至少5%的高流动性资产以应对紧急需求。")
        flags.append("liquidity_insufficient")

    # Fixed-income check for conservative/stable
    if risk_level in ("conservative", "stable") and fixed_income_total < 30.0:
        issues.append(f"固收类资产占比较低（{fixed_income_total:.0f}%），与保守/稳健风险偏好不匹配。")
        directions.append("可考虑提高短债、债券或偏债混合类资产的配置权重。")
        flags.append("fixed_income_underweight")

    return {
        "issues": issues,
        "adjustment_directions": directions,
        "risk_flags": flags,
    }
