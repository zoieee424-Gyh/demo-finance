"""
Sparse (keyword/lexical) retriever — architectural placeholder.

Design:
  - SparseRetriever: abstract interface for sparse/lexical retrieval.
  - BM25Retriever: placeholder shell for future BM25 over document chunks.

Current status:
  - BM25Retriever.search() returns an empty list — no real BM25 is
    implemented in this phase.
  - The shell exists so that HybridRetriever and downstream code can
    reference the abstraction without compile-time errors.

Future:
  - Implement a real BM25 scorer using a pre-built inverted index over
    ingested document chunks (stored in MySQL or in-memory).
  - Tokenizer: Jieba (Chinese) or language-agnostic whitespace/n-gram.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.consultation import Source

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SparseRetriever — abstract base
# ═══════════════════════════════════════════════════════════════════

class SparseRetriever:
    """Abstract interface for sparse/lexical retrieval.

    All sparse retrievers (BM25, TF-IDF, etc.) must implement search().
    """

    def search(
        self,
        query: str,
        collection_names: list[str],
        top_k: int = 5,
    ) -> list[Source]:
        """Return a ranked list of Source objects.

        Args:
            query: Raw search text.
            collection_names: Target knowledge collections.
            top_k: Maximum number of results.

        Returns:
            List of Source objects, sorted by relevance (descending).
        """
        raise NotImplementedError(
            "SparseRetriever.search() is abstract — "
            "use BM25Retriever or a concrete subclass."
        )


# ═══════════════════════════════════════════════════════════════════
# BM25Retriever — placeholder shell
# ═══════════════════════════════════════════════════════════════════

class BM25Retriever(SparseRetriever):
    """Placeholder BM25 sparse retriever.

    Current behavior:
      - search() always returns an empty list.
      - No crash, no dependency, no real scoring.

    Future implementation plan:
      - Load document chunks from MySQL metadata store.
      - Build an in-memory inverted index (token → doc frequency map).
      - Score using standard BM25 formula: IDF × TF × (k1+1) / (TF + k1×(1−b+b×doc_len/avg_len)).
      - Tokenizer: Jieba for Chinese, plus whitespace/n-gram fallback.
    """

    def __init__(self, metadata_store: Any = None) -> None:
        """
        Args:
            metadata_store: Optional KnowledgeMetadataStore for loading
                document chunks. Not used in placeholder; reserved for
                future real implementation.
        """
        self._metadata_store = metadata_store
        # Stateless for now — no index built.
        logger.info(
            "BM25Retriever initialized in placeholder mode — "
            "search() returns empty results."
        )

    def search(
        self,
        query: str,
        collection_names: list[str],
        top_k: int = 5,
    ) -> list[Source]:
        """Placeholder search — returns empty list.

        TODO: Implement BM25 scoring over ingested document chunks.
              - Fetch chunks from metadata_store filtered by collection_names.
              - Tokenize query and documents.
              - Compute BM25 score for each (query, document) pair.
              - Return top_k Source objects by score.
        """
        _ = (query, collection_names, top_k, self._metadata_store)
        logger.debug(
            "BM25Retriever.search() called in placeholder mode "
            "(query=%s, collections=%s) — returning empty list.",
            query[:80], collection_names,
        )
        return []


# ═══════════════════════════════════════════════════════════════════
# RetrievalMode utility
# ═══════════════════════════════════════════════════════════════════

def get_retrieval_mode() -> str:
    """Return the currently configured retrieval mode.

    Reads FIN_AGENT_RETRIEVAL_MODE from environment; defaults to
    "semantic" if unset or invalid.
    """
    from app.core.config import settings
    return settings.retrieval_mode
