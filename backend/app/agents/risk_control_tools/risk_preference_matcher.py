"""
RiskPreferenceMatcher（风险偏好匹配器）

Checks if portfolio risk aligns with user's stated risk preference.
Rule-based — no LLM.
"""

from __future__ import annotations

# ── Risk bands: max recommended equity % per risk level ──────────

_RISK_BANDS: dict[str, dict[str, float]] = {
    "conservative": {"max_equity": 20, "max_risky": 25, "min_cash": 15},
    "stable": {"max_equity": 40, "max_risky": 45, "min_cash": 10},
    "balanced": {"max_equity": 60, "max_risky": 65, "min_cash": 5},
    "aggressive": {"max_equity": 85, "max_risky": 90, "min_cash": 2},
}

_EQUITY = {"权益类", "股票类", "宽基指数类", "宽基指数基金类", "偏股混合基金类",
            "行业指数类", "A股", "港股", "美股", "全球权益类"}
_ALT = {"另类资产", "黄金", "商品", "REITs", "私募股权", "数字资产"}
_CASH = {"现金及货币类", "货币基金类"}


def match(*, holdings: list[dict] | None = None, risk_preference: str = "", **kwargs) -> dict:
    """Match portfolio risk against user's risk preference.

    Args:
        holdings: Portfolio holdings.
        risk_preference: User's risk preference (conservative/stable/balanced/aggressive).

    Returns:
        dict with match_status, mismatch_reasons, allowed_risk_band.
    """
    items = holdings or []
    pref = risk_preference or "balanced"
    band = _RISK_BANDS.get(pref, _RISK_BANDS["balanced"])

    equity_pct = sum(
        h["ratio"] for h in items if h.get("asset_class", "") in _EQUITY
    )
    alt_pct = sum(
        h["ratio"] for h in items if h.get("asset_class", "") in _ALT
    )
    cash_pct = sum(
        h["ratio"] for h in items if h.get("asset_class", "") in _CASH
    )
    risky_total = equity_pct + alt_pct

    mismatch_reasons: list[str] = []

    if equity_pct > band["max_equity"]:
        mismatch_reasons.append(
            f"权益类占比{equity_pct:.0f}%，超过{pref}风险偏好下"
            f"权益上限{band['max_equity']:.0f}%，组合风险已超出用户承受范围。"
        )

    if risky_total > band["max_risky"]:
        mismatch_reasons.append(
            f"高风险资产（权益+另类）合计{risky_total:.0f}%，"
            f"超过{pref}偏好上限{band['max_risky']:.0f}%。"
        )

    if cash_pct < band["min_cash"]:
        mismatch_reasons.append(
            f"现金类占比{cash_pct:.0f}%，低于{pref}偏好建议下限"
            f"{band['min_cash']:.0f}%，流动性缓冲不足。"
        )

    if mismatch_reasons:
        match_status = "mismatch"
    elif equity_pct <= band["max_equity"] * 0.6:
        match_status = "conservative_vs_preference"
    else:
        match_status = "match"

    return {
        "match_status": match_status,
        "risk_preference": pref,
        "equity_pct": equity_pct,
        "risky_total": risky_total,
        "cash_pct": cash_pct,
        "mismatch_reasons": mismatch_reasons,
        "allowed_risk_band": {
            "max_equity": band["max_equity"],
            "max_risky": band["max_risky"],
            "min_cash": band["min_cash"],
        },
    }
