"""
RiskInputParser（风险输入解析器）

Parses user question and optional holdings/risk_preference/financial_text
into structured risk review context. Rule-based — no LLM.
"""

from __future__ import annotations


def parse(*, question: str, user_profile: dict | None = None, **kwargs) -> dict:
    """Parse risk review inputs from user profile.

    Args:
        question: User's risk question.
        user_profile: Optional dict with holdings, risk_preference,
                      financial_text, liquidity_need.

    Returns:
        dict with review_scope, parsed_holdings, risk_preference, etc.
    """
    profile = user_profile or {}
    holdings = profile.get("holdings", [])
    risk_preference = profile.get("risk_preference", "")
    financial_text = profile.get("financial_text", "")
    liquidity_need = profile.get("liquidity_need", "")

    missing: list[str] = []

    # ── Validate holdings ──────────────────────────────────────
    parsed_holdings = []
    total = 0.0
    for h in holdings:
        if isinstance(h, dict):
            asset_class = h.get("asset_class", "未分类")
            ratio = float(h.get("ratio", 0))
            total += ratio
            parsed_holdings.append({"asset_class": asset_class, "ratio": ratio})

    if not parsed_holdings:
        missing.append("holdings")

    # ── Risk preference ────────────────────────────────────────
    valid_prefs = {"conservative", "stable", "balanced", "aggressive", "low", "medium", "high"}
    mapped_pref = _map_risk_preference(risk_preference)
    if not risk_preference:
        missing.append("risk_preference")
    elif risk_preference not in valid_prefs and mapped_pref is None:
        missing.append("valid_risk_preference")

    # ── Liquidity need ─────────────────────────────────────────
    if not liquidity_need:
        missing.append("liquidity_need")

    # ── Scope ──────────────────────────────────────────────────
    scope_parts = []
    if parsed_holdings:
        scope_parts.append("持仓组合审查")
    if financial_text:
        scope_parts.append("财务质量审查")
    if "风险" in question or "风控" in question:
        scope_parts.append("综合风险评估")
    scope_parts.append("风险缓释建议")

    return {
        "review_scope": "；".join(scope_parts),
        "parsed_holdings": parsed_holdings,
        "total_ratio": total,
        "risk_preference": mapped_pref or risk_preference or "unknown",
        "financial_text_available": bool(financial_text and len(financial_text) >= 10),
        "financial_text": financial_text,
        "liquidity_need": liquidity_need or "unknown",
        "missing_fields": missing,
    }


def _map_risk_preference(pref: str) -> str | None:
    """Map various risk preference labels to canonical values."""
    mapping = {
        "low": "conservative",
        "conservative": "conservative",
        "stable": "stable",
        "balanced": "balanced",
        "high": "aggressive",
        "aggressive": "aggressive",
    }
    return mapping.get(pref)
