"""
ComplianceInputParser（合规输入解析器）

Parses user content_to_review, scenario, audience, business_type.
Rule-based — no LLM.
"""

def parse(*, question: str, user_profile: dict | None = None, **kwargs) -> dict:
    profile = user_profile or {}
    content = profile.get("content_to_review", "")
    scenario = profile.get("scenario", "")
    audience = profile.get("audience", "retail_investor")
    business_type = profile.get("business_type", "")

    missing: list[str] = []
    if not content and question:
        content = question
    if not content or len(content.strip()) < 5:
        missing.append("content_to_review")
    if not scenario:
        missing.append("scenario")

    return {
        "review_content": content.strip(),
        "scenario": scenario or "unspecified",
        "audience": audience,
        "business_type": business_type or "unspecified",
        "missing_fields": missing,
    }
