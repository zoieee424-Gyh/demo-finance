"""
RegulatoryBasisMatcher（监管依据匹配器）

Maps violation flags to regulatory principles.
Rule-based — no LLM.
Does NOT output formal legal conclusions.
"""

_PRINCIPLES = {
    "个股推荐": {
        "principle": "禁止无资质荐股",
        "basis_note": "证券投资咨询业务需持牌经营，非持牌机构/个人不得向公众提供个股推荐建议。",
    },
    "目标价预测": {
        "principle": "禁止误导性预测",
        "basis_note": "对证券价格做出确定性预测可能构成误导性陈述，需基于充分客观数据并附风险提示。",
    },
    "收益承诺": {
        "principle": "禁止收益承诺",
        "basis_note": "不得以任何形式向投资者承诺收益或承担损失，不得使用'保本''稳赚'等误导性表述。",
    },
    "确定性预测": {
        "principle": "禁止误导性预测",
        "basis_note": "对市场或证券未来走势做出确定性判断可能违反信息披露和反欺诈相关规定。",
    },
    "交易指令": {
        "principle": "适当性管理与风险揭示",
        "basis_note": "提供具体交易操作建议需评估投资者适当性并充分揭示风险。",
    },
    "投资评级": {
        "principle": "禁止无资质荐股",
        "basis_note": "发布证券评级需具备相应资质，非持牌主体出具投资评级可能违规。",
    },
    "夸大宣传": {
        "principle": "禁止误导性宣传",
        "basis_note": "金融营销宣传不得含有虚假、夸大或引人误导的内容。",
    },
    "股票代码": {
        "principle": "禁止无资质荐股",
        "basis_note": "明确指向具体证券代码可能构成荐股行为，需具备相应资质。",
    },
}

def match(*, prohibited_flags: list | None = None, suitability_findings: list | None = None,
          missing_disclosures: list | None = None, **kwargs) -> dict:
    flags = prohibited_flags or []
    principles: list[dict] = []
    seen = set()

    for f in flags:
        cat = f.get("category", "") if isinstance(f, dict) else str(f)
        if cat in _PRINCIPLES and cat not in seen:
            seen.add(cat)
            principles.append(_PRINCIPLES[cat])

    # Add suitability-based principles
    if suitability_findings and len(suitability_findings) > 0:
        if "适当性管理" not in seen:
            seen.add("适当性管理")
            principles.append({
                "principle": "投资者适当性管理",
                "basis_note": "向投资者推荐产品或服务前应评估其风险承受能力和投资经验。",
            })

    # Add disclosure-based principles
    missing = missing_disclosures or []
    if missing and "信息披露充分性" not in seen:
        seen.add("信息披露充分性")
        principles.append({
            "principle": "信息披露充分性",
            "basis_note": "金融信息传播应包含充分的风险提示、数据来源和适用边界，不得选择性披露。",
        })

    return {
        "regulatory_principles": principles,
        "principle_count": len(principles),
        "basis_notes": [p["basis_note"] for p in principles],
    }
