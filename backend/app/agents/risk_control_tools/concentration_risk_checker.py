"""
ConcentrationRiskChecker（组合集中度检测器）

Checks portfolio concentration risks.
Rule-based — no LLM.
"""

from __future__ import annotations

# ── Asset classification sets ────────────────────────────────────

_EQUITY_ASSETS = {"权益类", "股票类", "宽基指数类", "宽基指数基金类", "偏股混合基金类",
                   "行业指数类", "A股", "港股", "美股", "全球权益类"}
_BOND_ASSETS = {"债券类", "国债/利率债类", "信用债类", "可转债类", "偏债混合基金类"}
_CASH_ASSETS = {"现金及货币类", "货币基金类"}
_ALTERNATIVE_ASSETS = {"另类资产", "黄金", "商品", "REITs", "私募股权", "数字资产"}


def check(*, holdings: list[dict] | None = None, **kwargs) -> dict:
    """Check portfolio concentration risks.

    Args:
        holdings: List of {"asset_class": str, "ratio": float}.

    Returns:
        dict with concentration_level, issues, concentration_flags.
    """
    items = holdings or []
    issues: list[str] = []
    flags: list[str] = []

    equity_pct = sum(
        h["ratio"] for h in items
        if h.get("asset_class", "") in _EQUITY_ASSETS
    )
    bond_pct = sum(
        h["ratio"] for h in items
        if h.get("asset_class", "") in _BOND_ASSETS
    )
    cash_pct = sum(
        h["ratio"] for h in items
        if h.get("asset_class", "") in _CASH_ASSETS
    )
    alt_pct = sum(
        h["ratio"] for h in items
        if h.get("asset_class", "") in _ALTERNATIVE_ASSETS
    )

    # ── Single asset class > 50% ───────────────────────────────
    for h in items:
        cls_name = h.get("asset_class", "")
        ratio = float(h.get("ratio", 0))
        if ratio > 50:
            issues.append(
                f"{cls_name}占比{ratio:.0f}%，超过50%集中度警戒线，"
                "单一类别过于集中，建议分散配置以降低个体风险。"
            )
            flags.append(f"single_class_concentration:{cls_name}")

    # ── Equity overweight ───────────────────────────────────────
    if equity_pct > 70:
        issues.append(
            f"权益类合计占比{equity_pct:.0f}%，组合波动风险较高，"
            "市场下行时可能面临较大回撤。"
        )
        flags.append("equity_overweight")
    elif equity_pct > 50:
        flags.append("equity_elevated")

    # ── Cash too low ────────────────────────────────────────────
    if cash_pct < 5:
        issues.append(
            f"现金及货币类仅占比{cash_pct:.0f}%，流动性储备不足，"
            "应急资金或市场机会出现时可能缺乏灵活应对空间。"
        )
        flags.append("cash_insufficient")

    # ── Equity + alternatives too high ──────────────────────────
    risky_total = equity_pct + alt_pct
    if risky_total > 80:
        issues.append(
            f"权益+另类资产合计占比{risky_total:.0f}%，超过80%高风险资产阈值，"
            "组合防御性较弱，极端市场环境下可能出现较大亏损。"
        )
        flags.append("high_risk_asset_overload")

    # ── Determine level ─────────────────────────────────────────
    if len(issues) >= 3:
        level = "high"
    elif len(issues) >= 1:
        level = "moderate"
    else:
        level = "low"

    return {
        "concentration_level": level,
        "equity_pct": equity_pct,
        "bond_pct": bond_pct,
        "cash_pct": cash_pct,
        "alt_pct": alt_pct,
        "issues": issues,
        "concentration_flags": flags,
    }
