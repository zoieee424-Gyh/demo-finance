"""
EducationEvidenceBuilder（投教证据构建器）

Builds education evidence pack from RAG sources.
Prioritizes education_knowledge, supplements with risk/compliance knowledge.
NEVER fabricates document content.

Rule-based — no LLM.
"""

from __future__ import annotations


def build(*, sources: list, **kwargs) -> dict:
    """Build education evidence from RAG sources.

    Args:
        sources: List of Source objects from KnowledgeRetriever.

    Returns:
        dict with evidence_summary, items, has_education, has_risk,
             has_compliance, low_confidence.
    """
    from app.schemas.consultation import Source

    normalized: list[Source] = []
    for s in (sources or []):
        try:
            if isinstance(s, Source):
                normalized.append(s)
            elif isinstance(s, dict):
                normalized.append(Source.model_validate(s))
        except Exception:
            # Skip invalid source entries — don't crash the tool
            pass

    if not normalized:
        return {
            "evidence_summary": "当前知识库中暂无相关参考资料。以下内容基于通用金融教育知识，请以监管机构和权威投教平台发布的材料为准。",
            "items": [],
            "has_education": False,
            "has_risk": False,
            "has_compliance": False,
            "low_confidence": True,
            "top_titles": [],
        }

    # Classify sources
    edu_items: list[dict] = []
    risk_items: list[dict] = []
    comp_items: list[dict] = []
    advisory_items: list[dict] = []

    for src in normalized:
        st = getattr(src, "source_type", "") or ""
        item = {
            "title": src.title,
            "source_type": st,
            "confidence": getattr(src, "confidence", None),
            "evidence_type": _map_source_to_evidence_type(st),
            "usage_hint": _get_usage_hint(st),
        }
        if "education" in st:
            edu_items.append(item)
        elif "risk" in st:
            risk_items.append(item)
        elif "compliance" in st:
            comp_items.append(item)
        else:
            advisory_items.append(item)

    # Priority: education > risk > compliance > advisory
    all_items = edu_items + risk_items + comp_items + advisory_items

    # Determine confidence
    top_conf = all_items[0].get("confidence") if all_items else None
    low_confidence = top_conf is None or top_conf < 0.45

    # Build summary
    if low_confidence and not all_items:
        summary = "当前知识库中暂无与问题直接相关的参考资料，以下内容基于通用金融教育知识。"
    elif low_confidence:
        summary = "检索到部分相关参考资料，但置信度有限。以下内容结合参考资料和通用知识，请以权威来源为准。"
    else:
        summary = f"基于{len(all_items)}条参考资料生成教育内容，详见参考依据章节。"

    return {
        "evidence_summary": summary,
        "items": all_items[:5],  # Top 5
        "has_education": len(edu_items) > 0,
        "has_risk": len(risk_items) > 0,
        "has_compliance": len(comp_items) > 0,
        "low_confidence": low_confidence,
        "top_titles": [item["title"] for item in all_items[:3]],
    }


def _map_source_to_evidence_type(source_type: str) -> str:
    """Map source_type to education evidence type."""
    if "education" in source_type:
        return "education"
    if "risk" in source_type:
        return "risk_reference"
    if "compliance" in source_type:
        return "compliance_reference"
    if "advisory" in source_type:
        return "advisory_reference"
    return "general_reference"


def _get_usage_hint(source_type: str) -> str:
    """Provide usage hint for the evidence item."""
    if "education" in source_type:
        return "可作为核心投教内容参考"
    if "risk" in source_type:
        return "可作为风险教育部分参考"
    if "compliance" in source_type:
        return "可作为合规边界说明参考"
    return "可作为辅助参考"
