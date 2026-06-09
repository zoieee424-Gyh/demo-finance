"""
Hybrid retriever — combines semantic (Chroma) and sparse (BM25) retrieval.

Design:
  - Wraps a semantic retriever (ChromaStore or KnowledgeRetriever) and an
    optional sparse retriever (BM25Retriever).
  - Merges results from both sources with deduplication.
  - Reserved for future Reciprocal Rank Fusion (RRF) or weighted sum.

Current status (placeholder):
  - sparse results are empty (BM25Retriever is a placeholder).
  - merge_results() returns semantic results as-is.
  - Production behavior is identical to semantic-only retrieval.

Future:
  - Implement real BM25 scoring.
  - Add RRF or weighted fusion with configurable weights.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.consultation import Source

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrate semantic + sparse retrieval with result fusion.

    Usage:
        hr = HybridRetriever(
            semantic_retriever=chroma_store,       # ChromaStore
            sparse_retriever=BM25Retriever(),       # placeholder
            semantic_weight=0.7,
            sparse_weight=0.3,
            top_k=5,
        )
        sources = hr.search("query", ["advisory_knowledge"], top_k=5)
    """

    def __init__(
        self,
        semantic_retriever: Any = None,
        sparse_retriever: Any = None,
        semantic_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            semantic_retriever: ChromaStore-compatible object with
                search(query, collection_names, top_k) → list[dict].
            sparse_retriever: SparseRetriever-compatible object.
                Defaults to BM25Retriever() if None.
            semantic_weight: Weight for semantic scores in fusion
                (reserved; currently unused).
            sparse_weight: Weight for sparse scores in fusion
                (reserved; currently unused).
            top_k: Default number of results.
        """
        self._semantic = semantic_retriever
        if sparse_retriever is None:
            from app.rag.sparse_retriever import BM25Retriever
            sparse_retriever = BM25Retriever()
        self._sparse = sparse_retriever
        self._semantic_weight = semantic_weight
        self._sparse_weight = sparse_weight
        self._top_k = top_k

        logger.info(
            "HybridRetriever initialized (semantic_weight=%.1f, sparse_weight=%.1f, "
            "top_k=%d). Sparse backend is placeholder — hybrid ≈ semantic.",
            semantic_weight, sparse_weight, top_k,
        )

    def search(
        self,
        query: str,
        collection_names: list[str],
        top_k: int | None = None,
    ) -> list[Source]:
        """Run semantic + sparse search and merge results.

        Args:
            query: User query text.
            collection_names: Target Chroma collections.
            top_k: Max results (defaults to self._top_k).

        Returns:
            Merged, deduplicated list of Source objects.
        """
        k = top_k or self._top_k

        # ── Semantic search ───────────────────────────────────
        semantic_sources: list[Source] = []
        if self._semantic is not None:
            try:
                raw = self._semantic.search(query, collection_names, top_k=k)
                semantic_sources = _chroma_results_to_sources(raw)
            except Exception as exc:
                logger.warning("Semantic search failed: %s", exc)

        # ── Sparse search ─────────────────────────────────────
        sparse_sources: list[Source] = []
        try:
            sparse_sources = self._sparse.search(query, collection_names, top_k=k)
        except Exception as exc:
            logger.warning("Sparse search failed: %s", exc)

        # ── Merge ─────────────────────────────────────────────
        return self._merge_results(semantic_sources, sparse_sources, k)

    def _merge_results(
        self,
        semantic: list[Source],
        sparse: list[Source],
        top_k: int,
    ) -> list[Source]:
        """Merge and deduplicate semantic + sparse results.

        Current strategy (placeholder):
          - Semantic results first (preserve order).
          - Append sparse results, deduplicate by title.
          - Truncate to top_k.

        Future: Reciprocal Rank Fusion (RRF) or weighted score fusion.
          - RRF: score(doc) = Σ 1 / (k + rank_i) across retrievers.
          - Weighted: score(doc) = w_sem × sem_score + w_sparse × bm25_score.
        """
        # TODO: implement RRF or weighted fusion
        merged: list[Source] = []
        seen_titles: set[str] = set()

        # Semantic first (preserve ranking order)
        for src in semantic:
            title = src.title
            if title and title not in seen_titles:
                seen_titles.add(title)
                merged.append(src)

        # Sparse append (dedup)
        for src in sparse:
            title = src.title
            if title and title not in seen_titles:
                seen_titles.add(title)
                merged.append(src)

        logger.debug(
            "Hybrid merge: semantic=%d, sparse=%d, merged=%d (top_k=%d)",
            len(semantic), len(sparse), len(merged), top_k,
        )

        return merged[:top_k]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _chroma_results_to_sources(raw_results: list[dict]) -> list[Source]:
    """Convert Chroma search result dicts to Source objects."""
    sources: list[Source] = []
    for r in raw_results:
        title = r.get("title", "") or r.get("metadata", {}).get("title", "")
        if not title:
            continue
        sources.append(Source(
            title=title,
            source_type=r.get("source_type", "knowledge_base"),
            url=r.get("metadata", {}).get("url"),
            confidence=round(r.get("score", 0.7), 4),
        ))
    return sources
