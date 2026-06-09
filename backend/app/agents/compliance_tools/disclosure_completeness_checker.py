"""
DisclosureCompletenessChecker（信息披露完整性检查器）

Checks if required disclosures are present.
Rule-based — no LLM.
"""

_REQUIRED_DISCLOSURES = {
    "risk_disclaimer": ["投资有风险", "风险提示", "入市需谨慎", "市场有风险"],
    "data_source": ["数据来源", "来源", "依据", "参考"],
    "applicability_boundary": ["适用", "仅供参考", "不构成", "不视为"],
    "no_investment_advice": ["不构成投资建议", "不构成投资", "不视为投资"],
    "past_performance": ["过往业绩", "历史表现", "不代表未来"],
    "no_legal_opinion": ["不构成法律", "不构成正式法律", "非法律意见"],
}

def check(*, review_content: str = "", **kwargs) -> dict:
    text = review_content or ""
    missing: list[str] = []

    for key, patterns in _REQUIRED_DISCLOSURES.items():
        if not any(p in text for p in patterns):
            missing.append(key)

    if len(missing) <= 1:
        level = "adequate"
    elif len(missing) <= 3:
        level = "incomplete"
    else:
        level = "severely_lacking"

    return {
        "disclosure_level": level,
        "missing_disclosures": missing,
        "present_count": len(_REQUIRED_DISCLOSURES) - len(missing),
        "total_required": len(_REQUIRED_DISCLOSURES),
    }
