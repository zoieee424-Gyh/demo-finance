"""
ProfileAnalyzer（用户画像解析器）

Parses user question text and user_profile dict into a structured financial profile.
Rule-based for MVP — no LLM calls.

Output:
  {
    "income_level": "low" | "medium" | "high",
    "risk_preference": "conservative" | "stable" | "balanced" | "aggressive",
    "liquidity_need": "low" | "medium" | "high",
    "investment_experience": "beginner" | "experienced",
    "constraints": [...],
    "missing_fields": [...]
  }
"""
from __future__ import annotations

import re
from typing import Any


# ── Risk preference keyword mapping ─────────────────────────────

_RISK_KEYWORDS: dict[str, str] = {
    "保守": "conservative", "厌恶": "conservative", "不能亏": "conservative",
    "保本": "conservative", "稳稳": "conservative", "安全": "conservative",
    "稳健": "stable", "稳定": "stable", "平稳": "stable",
    "平衡": "balanced", "均衡": "balanced",
    "进取": "aggressive", "积极": "aggressive", "高风险": "aggressive",
    "高收益": "aggressive", "激进": "aggressive",
}

# ── Experience keyword mapping ───────────────────────────────────

_EXPERIENCE_KEYWORDS: dict[str, str] = {
    "新手": "beginner", "小白": "beginner", "刚入门": "beginner",
    "刚开始": "beginner", "没有经验": "beginner", "不懂": "beginner",
    "入门": "beginner", "第一次": "beginner", "初学": "beginner",
    "有经验": "experienced", "多年": "experienced", "老手": "experienced",
    "熟悉": "experienced", "做过": "experienced",
}

# ── Income extraction patterns ───────────────────────────────────

_INCOME_PATTERNS = [
    (re.compile(r"月薪\s*(\d+)\s*[万千]"), lambda m: _classify_income(float(m.group(1)) * 10000 if "万" in m.group(0) else float(m.group(1)) * 1000)),
    (re.compile(r"月入\s*(\d+)\s*[万千]"), lambda m: _classify_income(float(m.group(1)) * 10000 if "万" in m.group(0) else float(m.group(1)) * 1000)),
    (re.compile(r"年薪\s*(\d+)\s*[万千]"), lambda m: _classify_income(float(m.group(1)) * 10000 / 12 if "万" in m.group(0) else float(m.group(1)) * 1000 / 12)),
    (re.compile(r"年收入\s*(\d+)\s*[万千]"), lambda m: _classify_income(float(m.group(1)) * 10000 / 12 if "万" in m.group(0) else float(m.group(1)) * 1000 / 12)),
]


def _classify_income(monthly: float) -> str:
    if monthly < 8000:
        return "low"
    elif monthly < 25000:
        return "medium"
    else:
        return "high"


# ── Constraint extraction ────────────────────────────────────────

_CONSTRAINT_PATTERNS = [
    (re.compile(r"(\d+)\s*年[之以]?后.{0,6}(买房|购房|首付)"), lambda m: f"{m.group(1)}年后买房"),
    (re.compile(r"(买房|购房|首付)"), lambda m: "买房需求"),
    (re.compile(r"(养老|退休)"), lambda m: "养老计划"),
    (re.compile(r"(教育|上学|孩子.{0,5}学)"), lambda m: "教育支出"),
    (re.compile(r"(不能|无法|不可).{0,5}(亏损|亏|赔)"), lambda m: "不能承受亏损"),
    (re.compile(r"(应急|备用|急用)"), lambda m: "流动性需求高"),
]


def analyze(question: str, user_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyze user profile from question text and optional profile dict.

    Args:
        question: Raw user question.
        user_profile: Optional pre-filled profile dict (e.g. {"risk_preference": "low"}).

    Returns:
        Structured profile dict.
    """
    profile = user_profile or {}
    result: dict[str, Any] = {
        "income_level": "medium",
        "risk_preference": "stable",
        "liquidity_need": "medium",
        "investment_experience": "beginner",
        "constraints": [],
        "missing_fields": [],
    }
    combined_text = question

    # ── Risk preference ──────────────────────────────────────
    # First check explicit profile field
    if "risk_preference" in profile:
        rp_map = {"low": "conservative", "medium": "stable", "high": "aggressive"}
        result["risk_preference"] = rp_map.get(str(profile["risk_preference"]).lower(), "stable")
    else:
        # Scan question text
        for kw, level in _RISK_KEYWORDS.items():
            if kw in combined_text:
                result["risk_preference"] = level
                break

    # ── Investment experience ────────────────────────────────
    if "investment_experience" in profile:
        result["investment_experience"] = str(profile["investment_experience"]).lower()
    else:
        for kw, level in _EXPERIENCE_KEYWORDS.items():
            if kw in combined_text:
                result["investment_experience"] = level
                break

    # ── Income level ─────────────────────────────────────────
    if "income_level" in profile:
        result["income_level"] = str(profile["income_level"]).lower()
    elif "monthly_income" in profile:
        result["income_level"] = _classify_income(float(profile["monthly_income"]))
    else:
        for pattern, classifier in _INCOME_PATTERNS:
            m = pattern.search(combined_text)
            if m:
                result["income_level"] = classifier(m)
                break

    # ── Liquidity need ───────────────────────────────────────
    if "liquidity_need" in profile:
        result["liquidity_need"] = str(profile["liquidity_need"]).lower()
    else:
        # Short-term goals → high liquidity need
        if re.search(r"(买房|购房|首付|应急|备用|急用|短期)", combined_text):
            result["liquidity_need"] = "high"
        elif re.search(r"(养老|退休|教育|长期|增值)", combined_text):
            result["liquidity_need"] = "low"
        # Default: medium

    # ── Constraints ──────────────────────────────────────────
    seen_constraints: set[str] = set()
    for pattern, extractor in _CONSTRAINT_PATTERNS:
        m = pattern.search(combined_text)
        if m:
            constraint = extractor(m)
            if constraint not in seen_constraints:
                seen_constraints.add(constraint)
                result["constraints"].append(constraint)

    # If profile has explicit constraints, merge them
    if "constraints" in profile and isinstance(profile["constraints"], list):
        for c in profile["constraints"]:
            if c not in seen_constraints:
                seen_constraints.add(c)
                result["constraints"].append(str(c))

    # ── Missing fields ───────────────────────────────────────
    if result["income_level"] == "medium" and "income_level" not in profile and "monthly_income" not in profile:
        # Income was defaulted, not explicitly found
        has_income_clue = any(p[0].search(combined_text) for p in _INCOME_PATTERNS)
        if not has_income_clue:
            result["missing_fields"].append("income_level")

    if "risk_preference" not in profile:
        has_risk_clue = any(kw in combined_text for kw in _RISK_KEYWORDS)
        if not has_risk_clue:
            result["missing_fields"].append("risk_preference")

    if "investment_horizon" not in profile and not re.search(r"\d+\s*年|\d+\s*个?月|短期|中期|长期", combined_text):
        result["missing_fields"].append("investment_horizon")

    return result


def enrich_profile_with_llm(
    profile: dict[str, Any],
    question: str,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Optionally enrich a profile with LLM-suggested completions.

    This is an OPTIONAL post-processing step. If the LLM provider is not
    configured, the profile is returned unchanged.

    The LLM is ONLY allowed to suggest values for fields that are currently
    UNKNOWN or MISSING. It MUST NOT:
      - Override explicitly provided user_profile fields.
      - Determine risk level (that is RiskAssessor's job).
      - Recommend stocks, funds, or products.

    Args:
        profile: Profile dict from analyze().
        question: Original user question.
        provider: LLMProvider instance (or None).

    Returns:
        Profile dict with LLM-suggested completions merged (only for missing fields).
    """
    if provider is None:
        return profile

    if not hasattr(provider, "is_configured") or not provider.is_configured():
        return profile

    missing = profile.get("missing_fields", [])
    if not missing:
        return profile  # No missing fields to enrich

    # Build prompt
    from app.llm.prompts import build_profile_prompt
    import json

    prompt = build_profile_prompt(
        current_profile=str({
            k: v for k, v in profile.items()
            if k in ("income_level", "risk_preference", "investment_experience", "liquidity_need", "constraints")
        }),
        missing_fields=", ".join(missing),
        user_question=question,
    )

    raw = provider.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )

    # Parse LLM response — if it's a mock response, skip
    if raw.startswith("[LLM Mock]"):
        return profile

    try:
        # Try to extract JSON from the response
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            suggestions = json.loads(raw[json_start:json_end])
        else:
            return profile
    except (json.JSONDecodeError, ValueError):
        return profile

    # ── Merge suggestions for missing fields ONLY ────────────────
    # Map LLM output keys to profile keys
    _field_map = {
        "income_level": "income_level",
        "investment_experience": "investment_experience",
        "liquidity_need": "liquidity_need",
        "investment_horizon": "investment_horizon",
        "constraints": "constraints",
    }
    # IMPORTANT: risk_preference is NOT merged — it must come from
    # explicit user input or keyword matching, not LLM inference.

    allowed_values = {
        "income_level": {"low", "medium", "high"},
        "investment_experience": {"beginner", "experienced"},
        "liquidity_need": {"low", "medium", "high"},
        "investment_horizon": {"short", "medium", "long"},
    }

    for llm_key, profile_key in _field_map.items():
        if profile_key in missing and llm_key in suggestions:
            value = suggestions[llm_key]
            if value and value != "unknown":
                if profile_key == "constraints" and isinstance(value, list):
                    existing = set(profile.get("constraints", []))
                    for c in value:
                        if c not in existing:
                            existing.add(c)
                            profile["constraints"].append(c)
                    if profile_key in profile.get("missing_fields", []):
                        profile["missing_fields"].remove(profile_key)
                elif profile_key != "constraints":
                    if profile_key in allowed_values and value not in allowed_values[profile_key]:
                        continue
                    profile[profile_key] = value
                    if profile_key in profile.get("missing_fields", []):
                        profile["missing_fields"].remove(profile_key)

    return profile
