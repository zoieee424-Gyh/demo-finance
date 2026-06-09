"""GraphRAG architecture shell.

This module reserves the graph-based retrieval extension point without changing
the current Chroma semantic retrieval behavior.

Intended future use:
  - entity graph: company, fund, regulation, risk factor, financial indicator
  - relation graph: "company has risk", "rule restricts expression",
    "indicator belongs to report section"
  - community / path retrieval: retrieve related nodes through graph paths,
    then use the node evidence together with vector/BM25 results.

Current behavior:
  - no graph database dependency is introduced
  - search() returns an empty list when no graph_store is injected
  - if a graph_store is injected later, it may expose search(query, intent, top_k)
    and return Source objects or dicts compatible with Source
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.consultation import Source


@dataclass(frozen=True)
class GraphRAGQueryContext:
    """Minimal query context reserved for future graph expansion."""

    query: str
    intent: str = "education"
    entity_hints: tuple[str, ...] = ()
    relation_hints: tuple[str, ...] = ()


class GraphRAGRetriever:
    """Placeholder GraphRAG retriever.

    The class is intentionally dependency-free.  It lets the rest of the RAG
    chain know where GraphRAG will be plugged in, while keeping the MVP stable.
    """

    def __init__(self, graph_store: Any = None) -> None:
        self.graph_store = graph_store

    def search(
        self,
        query: str,
        *,
        intent: str = "education",
        top_k: int = 5,
    ) -> list[Source]:
        """Search graph evidence.

        Returns an empty list until a real graph_store is provided.  A future
        graph_store can return either Source objects or dicts with title,
        source_type, url and confidence.
        """
        if not query.strip() or self.graph_store is None:
            return []

        search = getattr(self.graph_store, "search", None)
        if not callable(search):
            return []

        try:
            raw_results = search(query=query, intent=intent, top_k=top_k)
        except Exception:
            return []

        return self._to_sources(raw_results, top_k=top_k)

    @staticmethod
    def build_context(query: str, intent: str = "education") -> GraphRAGQueryContext:
        """Build a lightweight context object for future entity extraction."""
        return GraphRAGQueryContext(query=query, intent=intent)

    @staticmethod
    def _to_sources(raw_results: Any, *, top_k: int) -> list[Source]:
        sources: list[Source] = []
        if not isinstance(raw_results, list):
            return sources

        for item in raw_results[:top_k]:
            if isinstance(item, Source):
                sources.append(item)
                continue
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or ""
            if not title:
                continue
            sources.append(
                Source(
                    title=title,
                    source_type=item.get("source_type", "graph_knowledge"),
                    url=item.get("url"),
                    confidence=float(item.get("confidence", 0.6)),
                )
            )
        return sources
