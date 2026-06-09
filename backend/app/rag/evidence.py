"""
Evidence data structures — convert RAG Source lists into structured
evidence for agent answer generation.

Design:
  - Pure rule-based, no LLM calls.
  - Maps source_type → evidence_type for domain-aware routing.
  - Provides usage hints for each evidence type.
  - Detects low-confidence scenarios for answer downgrading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.consultation import Source


# ── source_type → evidence_type mapping ─────────────────────────────

_SOURCE_TO_EVIDENCE: dict[str, str] = {
    "advisory_knowledge": "advisory",
    "compliance_knowledge": "compliance",
    "education_knowledge": "education",
    "risk_knowledge": "risk",
    "financial_report_knowledge": "advisory",
    "investment_knowledge": "advisory",
    "asset_allocation": "advisory",
    "risk_models": "risk",
    "risk_indicators": "risk",
    "risk_management": "risk",
    "market_indicators": "risk",
    "company_financials": "risk",
    "news_sentiment": "risk",
    "regulations": "compliance",
    "legal_documents": "compliance",
    "compliance_checklists": "compliance",
    "financial_education": "education",
    "glossary": "education",
    "tutorials": "education",
    "financial_reports": "advisory",
    "company_filings": "advisory",
    "industry_benchmarks": "advisory",
    "market_data": "advisory",
    "knowledge_base": "unknown",
}

# ── evidence_type → usage hint ──────────────────────────────────────

_USAGE_HINTS: dict[str, str] = {
    "advisory": "可作为资产配置/定投建议的依据参考",
    "risk": "可作为风险识别与风险管理的依据参考",
    "compliance": "可作为合规边界与风险揭示的依据参考",
    "education": "可作为金融概念解释的依据参考",
    "unknown": "可作为一般性知识参考",
}

# ── Low-confidence threshold ────────────────────────────────────────

_LOW_CONFIDENCE_THRESHOLD = 0.45


@dataclass
class EvidenceItem:
    """A single evidence item derived from a RAG Source."""

    title: str
    source_type: str
    confidence: float | None
    url: str | None
    evidence_type: str          # advisory / risk / compliance / education / unknown
    summary: str                # Short label derived from title + type
    usage_hint: str             # How this evidence should be used in an answer


@dataclass
class EvidencePack:
    """Structured evidence bundle for agent answer generation."""

    items: list[EvidenceItem]
    by_type: dict[str, list[EvidenceItem]] = field(default_factory=dict)
    has_advisory: bool = False
    has_risk: bool = False
    has_compliance: bool = False
    has_education: bool = False
    top_titles: list[str] = field(default_factory=list)
    low_confidence: bool = False


def _map_evidence_type(source_type: str) -> str:
    """Map a source_type string to an evidence_type category."""
    # Direct match
    if source_type in _SOURCE_TO_EVIDENCE:
        return _SOURCE_TO_EVIDENCE[source_type]
    # Prefix match (e.g. "risk_*" → risk)
    for prefix, ev_type in [
        ("risk_", "risk"),
        ("compliance_", "compliance"),
        ("education_", "education"),
        ("advisory_", "advisory"),
        ("financial_report_", "advisory"),
        ("investment_", "advisory"),
        ("asset_", "advisory"),
        ("market_", "risk"),
        ("regulatory_", "compliance"),
    ]:
        if source_type.startswith(prefix):
            return ev_type
    return "unknown"


def _build_short_summary(title: str, evidence_type: str) -> str:
    """Build a one-line evidence summary from title + evidence_type.

    Does NOT fabricate document content — only uses the title and type.
    """
    type_labels: dict[str, str] = {
        "advisory": "投顾知识",
        "risk": "风控知识",
        "compliance": "合规知识",
        "education": "金融科普",
        "unknown": "参考资料",
    }
    label = type_labels.get(evidence_type, "参考资料")
    return f"《{title}》（{label}）"


def build_evidence_pack(
    sources: list[Source],
    max_items: int = 6,
) -> EvidencePack:
    """Convert a list of RAG Sources into a structured EvidencePack.

    Args:
        sources: Source objects from KnowledgeRetriever.retrieve().
        max_items: Maximum number of evidence items to include.

    Returns:
        EvidencePack with typed items, lookup helpers, and confidence flag.
    """
    items: list[EvidenceItem] = []
    seen_titles: set[str] = set()

    for src in sources:
        if len(items) >= max_items:
            break
        title = src.title
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)

        ev_type = _map_evidence_type(src.source_type)
        items.append(EvidenceItem(
            title=title,
            source_type=src.source_type,
            confidence=src.confidence,
            url=src.url,
            evidence_type=ev_type,
            summary=_build_short_summary(title, ev_type),
            usage_hint=_USAGE_HINTS.get(ev_type, _USAGE_HINTS["unknown"]),
        ))

    # Group by evidence_type
    by_type: dict[str, list[EvidenceItem]] = {}
    for item in items:
        by_type.setdefault(item.evidence_type, []).append(item)

    # Determine low_confidence. Missing confidence is treated as low
    # confidence because the answer cannot verify retrieval strength.
    top_confidence = sources[0].confidence if sources else None
    low_conf = (
        not sources
        or top_confidence is None
        or top_confidence < _LOW_CONFIDENCE_THRESHOLD
    )

    return EvidencePack(
        items=items,
        by_type=by_type,
        has_advisory=any(it.evidence_type == "advisory" for it in items),
        has_risk=any(it.evidence_type == "risk" for it in items),
        has_compliance=any(it.evidence_type == "compliance" for it in items),
        has_education=any(it.evidence_type == "education" for it in items),
        top_titles=[it.title for it in items],
        low_confidence=low_conf,
    )


def format_evidence_section(pack: EvidencePack) -> str:
    """Render an evidence section suitable for inclusion in agent answers.

    Returns a multi-line string with formatted evidence items and
    low-confidence warning (if applicable).
    """
    lines: list[str] = []

    if pack.low_confidence:
        lines.append("  ⚠ 当前参考资料置信度有限，以下建议仅作通用框架，请谨慎参考。")
        lines.append("")

    if not pack.items:
        lines.append("  当前无可用的知识库参考资料。")
        return "\n".join(lines)

    for i, item in enumerate(pack.items, 1):
        conf_str = f"{item.confidence:.0%}" if item.confidence is not None else "N/A"
        lines.append(f"  {i}. {item.summary}（置信度 {conf_str}）：{item.usage_hint}")

    return "\n".join(lines)


def format_evidence_inline(
    pack: EvidencePack,
    evidence_type: str | None = None,
    max_items: int = 2,
) -> str:
    """Render a short inline evidence reference for a specific section.

    Args:
        pack: The EvidencePack.
        evidence_type: Filter to this type (None = all types).
        max_items: Maximum items to reference.

    Returns:
        A short inline string like "依据：《资产配置基础原则》《风险等级匹配》".
    """
    candidates = pack.by_type.get(evidence_type, []) if evidence_type else pack.items
    if not candidates:
        return ""

    titles = [f"《{it.title}》" for it in candidates[:max_items]]
    return f"参考依据：{'、'.join(titles)}"
