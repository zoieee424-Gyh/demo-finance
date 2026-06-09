"""
LiquidityRiskAssessor（流动性风险评估器）

Assesses liquidity risk from portfolio structure and financial text.
Rule-based — no LLM.
"""

from __future__ import annotations

_CASH = {"现金及货币类", "货币基金类"}
_BOND = {"债券类", "国债/利率债类", "信用债类", "偏债混合基金类"}


def assess(
    *,
    holdings: list[dict] | None = None,
    liquidity_need: str = "",
    financial_text: str = "",
    **kwargs,
) -> dict:
    """Assess liquidity risk.

    Args:
        holdings: Portfolio holdings.
        liquidity_need: User's liquidity need (low/medium/high).
        financial_text: Optional financial report text.

    Returns:
        dict with liquidity_level, liquidity_findings, cashflow_pressure_flags.
    """
    items = holdings or []
    findings: list[str] = []
    flags: list[str] = []

    cash_pct = sum(
        h["ratio"] for h in items if h.get("asset_class", "") in _CASH
    )
    bond_pct = sum(
        h["ratio"] for h in items if h.get("asset_class", "") in _BOND
    )

    risk_score = 0

    # ── Cash position ──────────────────────────────────────────
    if cash_pct < 5:
        findings.append(f"现金类仅占比{cash_pct:.0f}%，极端情况下变现能力不足")
        flags.append("ultra_low_cash")
        risk_score += 2
    elif cash_pct < 10:
        findings.append(f"现金类占比{cash_pct:.0f}%，流动性储备偏紧")
        risk_score += 1

    # ── Liquidity need vs cash ─────────────────────────────────
    liq_req = {"high": 15, "medium": 8, "low": 3}.get(liquidity_need, 5)
    if cash_pct < liq_req:
        findings.append(
            f"流动性需求为{liquidity_need or '未明确'}，"
            f"建议现金占比不低于{liq_req}%，当前{cash_pct:.0f}%不足。"
        )
        flags.append("cash_below_liquidity_need")
        risk_score += 1

    # ── Liquid assets check ────────────────────────────────────
    liquid_pct = cash_pct + bond_pct * 0.7  # bonds partially liquid
    if liquid_pct < 20:
        findings.append(f"高流动性资产合计约{liquid_pct:.0f}%，整体流动性偏弱")
        flags.append("low_liquid_assets")
        risk_score += 1

    # ── Financial text cash flow checks ────────────────────────
    text = financial_text or ""
    has_negative_ocf = "经营现金流" in text and ("-3" in text or "- " in text or "为负" in text)
    has_high_debt = any(
        f"资产负债率{pct}%" in text for pct in ["70", "75", "80", "85", "90", "95"]
    ) or ("资产负债率" in text and "高" in text)

    if has_negative_ocf:
        findings.append("财报显示经营现金流为负，企业经营造血能力不足")
        flags.append("negative_ocf")
        risk_score += 2

    if has_high_debt:
        findings.append("资产负债率处于高位，偿债压力可能影响流动性")
        flags.append("high_debt_ratio")
        risk_score += 1

    # ── Determine level ────────────────────────────────────────
    if risk_score >= 4:
        level = "high_risk"
    elif risk_score >= 2:
        level = "moderate_risk"
    elif risk_score >= 1:
        level = "elevated"
    else:
        level = "adequate"

    return {
        "liquidity_level": level,
        "cash_pct": cash_pct,
        "liquid_pct": liquid_pct,
        "liquidity_findings": findings,
        "cashflow_pressure_flags": flags,
    }
