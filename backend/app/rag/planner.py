"""
Agentic RAG Planner — structured retrieval plan generation.

Design:
  - Produces query plans from an intent + ConsultationRequest.
  - Current strategy: Chroma semantic search only.
  - BM25 / hybrid fields are reserved in query dicts but NOT implemented.
  - No real LLM call — decomposition is rule-based.
  - Deterministic, testable, and dependency-free for MVP.

Interface contract (expected by ConsultationService):
  RagPlanner().build_queries(request: ConsultationRequest, intent: str) -> list[dict]
"""
from __future__ import annotations

from app.schemas.consultation import ConsultationRequest


# ── Intent → recommended source types ────────────────────────────

_INTENT_SOURCE_TYPES: dict[str, list[str]] = {
    "advisory":        ["investment_knowledge", "market_data", "asset_allocation"],
    "financial_report": ["financial_reports", "company_filings", "industry_benchmarks"],
    "risk_control":     ["risk_models", "market_indicators", "company_financials", "news_sentiment"],
    "compliance":       ["regulations", "legal_documents", "compliance_checklists"],
    "education":        ["financial_education", "glossary", "tutorials"],
}


def _extract_key_phrases(question: str) -> list[str]:
    """Extract potential key phrases via simple heuristics.

    Returns up to 3 phrases for use as focused search queries.
    """
    phrases: list[str] = []
    parts = question.replace("？", " ").replace("，", " ").replace(",", " ").replace("、", " ").split()
    candidates = [p for p in parts if len(p) >= 2]
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            phrases.append(c)
        if len(phrases) >= 3:
            break
    return phrases


class RagPlanner:
    """Generates structured retrieval query plans.

    Usage:
        planner = RagPlanner()
        queries = planner.build_queries(request, "advisory")
        # → [{"query": "...", "source_type": "...", "strategy": "semantic", ...}, ...]
    """

    def build_queries(self, request: ConsultationRequest, intent: str) -> list[dict]:
        """Build a list of query dicts for the retriever.

        Args:
            request: The incoming consultation request.
            intent: Classified intent label string.

        Returns:
            List of query dicts, each with:
              - query: search text
              - source_type: target knowledge domain
              - strategy: "semantic" (MVP)
              - bm25_weight: None (reserved)
              - semantic_weight: None (reserved)
        """
        source_types = _INTENT_SOURCE_TYPES.get(intent, ["general_knowledge"])
        phrases = _extract_key_phrases(request.question)
        queries: list[dict] = []

        # Query 0: full original question → primary source type
        queries.append({
            "query": request.question,
            "source_type": source_types[0],
            "strategy": "semantic",
            "bm25_weight": None,       # reserved for hybrid
            "semantic_weight": None,   # reserved for hybrid
        })

        # Queries 1+: entity-specific → secondary source types
        for i, phrase in enumerate(phrases):
            source_type = source_types[min(i + 1, len(source_types) - 1)]
            queries.append({
                "query": phrase,
                "source_type": source_type,
                "strategy": "semantic",
                "bm25_weight": None,
                "semantic_weight": None,
            })

        return queries
