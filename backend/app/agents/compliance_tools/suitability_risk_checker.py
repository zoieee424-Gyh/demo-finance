"""
SuitabilityRiskChecker（适当性风险检测器）

Checks if content is suitable for the intended audience.
Rule-based — no LLM.
"""

_HIGH_RISK_TERMS = {"股票", "个股", "期货", "期权", "私募", "杠杆", "高风险", "激进"}
_LOW_RISK_AUDIENCE = {"retail_investor", "普通投资者", "conservative", "stable", "beginner"}

def check(*, review_content: str = "", audience: str = "", scenario: str = "", **kwargs) -> dict:
    text = review_content or ""
    findings: list[str] = []
    risk_score = 0

    is_low_risk_audience = any(a in str(audience).lower() for a in _LOW_RISK_AUDIENCE)
    has_high_risk_content = any(t in text for t in _HIGH_RISK_TERMS)

    if is_low_risk_audience and has_high_risk_content:
        findings.append("内容涉及高风险产品或策略，但目标受众为普通/低风险投资者，存在适当性风险。")
        risk_score += 2

    if "recommend" in scenario.lower() or "推荐" in text or "建议" in text:
        if is_low_risk_audience:
            findings.append("向普通投资者提供具体产品建议时需充分揭示风险并评估适当性。")
            risk_score += 1

    if not findings:
        return {"suitability_level": "adequate", "suitability_findings": findings}

    level = "high_risk" if risk_score >= 2 else "elevated"
    return {"suitability_level": level, "suitability_findings": findings}
