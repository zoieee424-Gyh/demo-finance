"""
Knowledge Retriever — unified retrieval with Chroma + mock fallback.

Interface contract (expected by ConsultationService):
    KnowledgeRetriever().retrieve(queries: list[dict]) -> list[Source]

Design:
    - If Chroma is available and collections have data → semantic search.
    - Otherwise → fall back to intent-keyed mock sources.
    - Uses CollectionSelector to map intent → Chroma collections.
    - Confidence scores: real (1 - cosine_distance) for Chroma;
      simulated values for mock.
"""
from __future__ import annotations

from typing import Any

from app.schemas.consultation import Source
from app.rag.collection_selector import CollectionSelector
from app.rag.secondary_intent_detector import SecondaryIntentDetector


# ── Mock evidence store (fallback) ──────────────────────────────────

_MOCK_STORE: dict[str, list[dict]] = {
    "advisory": [
        {
            "title": "资产配置基础原则",
            "source_type": "investment_knowledge",
            "url": None,
            "confidence": 0.85,
        },
        {
            "title": "生命周期投资理论",
            "source_type": "asset_allocation",
            "url": None,
            "confidence": 0.78,
        },
    ],
    "financial_report": [
        {
            "title": "利润表核心指标解读",
            "source_type": "financial_reports",
            "url": None,
            "confidence": 0.90,
        },
        {
            "title": "现金流量表分析要点",
            "source_type": "financial_reports",
            "url": None,
            "confidence": 0.82,
        },
    ],
    "risk_control": [
        {
            "title": "Beneish M-Score 财务造假识别模型",
            "source_type": "risk_models",
            "url": None,
            "confidence": 0.92,
        },
        {
            "title": "Altman Z-Score 破产预测模型",
            "source_type": "risk_models",
            "url": None,
            "confidence": 0.88,
        },
    ],
    "compliance": [
        {
            "title": "证券法信息披露要求",
            "source_type": "regulations",
            "url": None,
            "confidence": 0.95,
        },
        {
            "title": "资管新规核心要点",
            "source_type": "regulations",
            "url": None,
            "confidence": 0.91,
        },
    ],
    "education": [
        {
            "title": "基金入门：什么是公募基金",
            "source_type": "financial_education",
            "url": None,
            "confidence": 0.88,
        },
        {
            "title": "PE与PB估值指标基础",
            "source_type": "glossary",
            "url": None,
            "confidence": 0.85,
        },
    ],
}

# Map intent → source_type strings for reverse lookup
_INTENT_SOURCE_MAP: dict[str, list[str]] = {
    "advisory":        ["investment_knowledge", "market_data", "asset_allocation"],
    "financial_report": ["financial_reports", "company_filings", "industry_benchmarks"],
    "risk_control":     ["risk_models", "market_indicators", "company_financials", "news_sentiment"],
    "compliance":       ["regulations", "legal_documents", "compliance_checklists"],
    "education":        ["financial_education", "glossary", "tutorials"],
}

# Map collection_name → canonical source_type for inference
_COLLECTION_TO_SOURCE_TYPE: dict[str, str] = {
    "advisory_knowledge": "advisory_knowledge",
    "compliance_knowledge": "compliance_knowledge",
    "education_knowledge": "education_knowledge",
    "risk_knowledge": "risk_knowledge",
    "financial_report_knowledge": "financial_report_knowledge",
}


def _infer_source_type_from_collection(collection_name: str) -> str:
    """Infer a canonical source_type from a Chroma collection name."""
    return _COLLECTION_TO_SOURCE_TYPE.get(collection_name, "")


class KnowledgeRetriever:
    """Unified knowledge retriever — Chroma-first with mock fallback.

    Usage:
        retriever = KnowledgeRetriever()
        # Optional: inject ChromaStore for real semantic search
        retriever = KnowledgeRetriever(chroma_store=store)
        sources = retriever.retrieve(queries)
        # → [Source(title="...", url=None, confidence=0.85), ...]
    """

    def __init__(
        self,
        chroma_store: Any = None,
        graph_store: Any = None,
        selector: Any = None,
        secondary_detector: Any = None,
    ) -> None:
        """
        Args:
            chroma_store: ChromaStore instance (or None for mock-only).
            selector: CollectionSelector instance (or None to create default).
            secondary_detector: SecondaryIntentDetector (or None to create default).
        """
        self._chroma_store = chroma_store
        self._graph_store = graph_store
        self._selector = selector or CollectionSelector()
        self._secondary_detector = secondary_detector or SecondaryIntentDetector()

    @property
    def has_chroma(self) -> bool:
        """Return True if Chroma is available and has data."""
        if self._chroma_store is None:
            return False
        try:
            collections = self._chroma_store.list_collections()
            return any(c.get("count", 0) > 0 for c in collections)
        except Exception:
            return False

    @property
    def retrieval_mode(self) -> str:
        """The currently configured retrieval mode (semantic/bm25/hybrid)."""
        from app.core.config import settings
        return settings.retrieval_mode

    @property
    def graphrag_enabled(self) -> bool:
        """Whether optional GraphRAG evidence expansion is enabled."""
        from app.core.config import settings
        return settings.graphrag_enabled

    def retrieve(self, queries: list[dict]) -> list[Source]:
        """Retrieve sources for the given queries.

        按 FIN_AGENT_RETRIEVAL_MODE 分派检索路径：
          - "semantic": 当前可用主路径，走 Chroma 语义检索。
          - "bm25":     稀疏检索壳子，当前返回空结果。
          - "hybrid":   混合检索壳子，当前等价于 semantic。

        只有 semantic/hybrid 在真实检索无结果时才回退 mock；bm25-only
        保持“空壳即空结果”，避免让调用方误以为 BM25 已可用。

        Args:
            queries: List of query dicts from RagPlanner.build_queries().

        Returns:
            List of Source objects, deduplicated by title.
        """
        mode = self.retrieval_mode

        # BM25 当前只是架构占位：显式选择 bm25 时返回空，暴露“未实现”状态。
        if mode == "bm25":
            return self._retrieve_bm25_placeholder(queries)

        # Hybrid 已预留融合入口；在 BM25 未实现前，它实际仍依赖 Chroma。
        if mode == "hybrid":
            sources = self._retrieve_hybrid(queries)
            if sources:
                return self._merge_sources(
                    sources,
                    self._retrieve_graphrag_placeholder(queries),
                )
            return self._retrieve_from_mock(queries)

        # 默认主路径：Chroma 语义检索。
        if self.has_chroma:
            sources = self._retrieve_from_chroma(queries)
            if sources:
                return self._merge_sources(
                    sources,
                    self._retrieve_graphrag_placeholder(queries),
                )

        # 本地展示兜底：无 Chroma 或无结果时仍给出可展示来源。
        return self._retrieve_from_mock(queries)

    # ── Mode-aware retrieval methods ──────────────────────────────

    def _retrieve_hybrid(self, queries: list[dict]) -> list[Source]:
        """Hybrid retrieval: semantic + sparse, merge with dedup.

        当前 BM25Retriever 为空壳，因此返回结果约等于 semantic。
        保留该入口是为了后续接入 RRF/加权融合时不改调用方。
        """
        intent = self._infer_intent(queries)
        query_text = " ".join(q.get("query", "") for q in queries)
        if not query_text.strip():
            return []

        from app.rag.hybrid_retriever import HybridRetriever

        hr = HybridRetriever(
            semantic_retriever=self._chroma_store,
            semantic_weight=0.7,
            sparse_weight=0.3,
        )
        collection_names = self._selector.get_collections_for_intents(
            intent,
            self._secondary_detector.detect(query_text, primary_intent=intent),
            self._chroma_store,
        )
        if not collection_names:
            return []

        top_k = min(len(queries) * 3, 10)
        return hr.search(query_text, collection_names, top_k=top_k)

    def _retrieve_bm25_placeholder(self, queries: list[dict]) -> list[Source]:
        """BM25-only 检索占位：真实 BM25 未实现前始终返回空。"""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "BM25 retrieval requested but not implemented — returning empty. "
            "Use semantic or hybrid for functional retrieval."
        )
        _ = queries
        return []

    def _retrieve_graphrag_placeholder(self, queries: list[dict]) -> list[Source]:
        """Optional GraphRAG evidence expansion shell.

        Current MVP behavior:
          - disabled by default via FIN_AGENT_GRAPHRAG_ENABLED=false
          - when enabled but no graph_store is injected, returns []
          - future graph_store can add entity/relation/community evidence
        """
        if not self.graphrag_enabled:
            return []

        query_text = " ".join(q.get("query", "") for q in queries)
        if not query_text.strip():
            return []

        from app.rag.graph_retriever import GraphRAGRetriever

        retriever = GraphRAGRetriever(graph_store=self._graph_store)
        return retriever.search(
            query_text,
            intent=self._infer_intent(queries),
            top_k=min(len(queries) * 2, 6),
        )

    @staticmethod
    def _merge_sources(primary: list[Source], extra: list[Source]) -> list[Source]:
        """Merge evidence sources by title while preserving primary order."""
        if not extra:
            return primary

        merged: list[Source] = []
        seen: set[str] = set()
        for source in [*primary, *extra]:
            title = source.title or ""
            if not title or title in seen:
                continue
            seen.add(title)
            merged.append(source)
        return merged

    # ── Chroma retrieval ────────────────────────────────────────────

    def _retrieve_from_chroma(self, queries: list[dict]) -> list[Source]:
        """Retrieve sources using Chroma semantic search with multi-intent
        collection selection (primary + secondary intents)."""
        intent = self._infer_intent(queries)

        # Combine all query texts into a single search string
        query_text = " ".join(q.get("query", "") for q in queries)
        if not query_text.strip():
            return []

        # Detect secondary intents for cross-domain retrieval
        secondary = self._secondary_detector.detect(query_text, primary_intent=intent)

        # Merge collections: primary first, then secondary
        collection_names = self._selector.get_collections_for_intents(
            intent, secondary, self._chroma_store
        )

        if not collection_names:
            return []

        top_k = min(len(queries) * 3, 10)  # ~3 results per query, max 10

        try:
            raw_results = self._chroma_store.search(
                query_text, collection_names, top_k=top_k
            )
        except Exception:
            return []

        if not raw_results:
            return []

        # 将 Chroma 原始结果标准化为前端/报告统一使用的 Source。
        sources: list[Source] = []
        seen_titles: set[str] = set()

        for r in raw_results:
            title = r.get("title", "") or r.get("metadata", {}).get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # 老数据可能没有 source_type，或只写了 generic knowledge_base；
            # 此时根据 collection_name 推断，减少 evidence 中的 unknown。
            raw_source_type = r.get("source_type", "")
            if not raw_source_type or raw_source_type == "knowledge_base":
                inferred = _infer_source_type_from_collection(
                    r.get("collection_name", "")
                )
                if inferred:
                    raw_source_type = inferred

            sources.append(Source(
                title=title,
                source_type=raw_source_type or "knowledge_base",
                url=r.get("metadata", {}).get("url"),
                confidence=round(r.get("score", 0.7), 4),
            ))

        return sources

    # ── Mock fallback ───────────────────────────────────────────────

    def _retrieve_from_mock(self, queries: list[dict]) -> list[Source]:
        """Legacy mock retrieval — returns canned sources by intent."""
        intent = self._infer_intent(queries)

        mock_items = _MOCK_STORE.get(intent, _MOCK_STORE["education"])
        seen: set[str] = set()
        sources: list[Source] = []

        for item in mock_items:
            title = item.get("title", "")
            if title and title not in seen:
                seen.add(title)
                sources.append(Source(
                    title=title,
                    source_type=item.get("source_type", "knowledge_base"),
                    url=item.get("url"),
                    confidence=item.get("confidence", 0.5),
                ))

        return sources

    # ── Intent inference ─────────────────────────────────────────────

    @staticmethod
    def _infer_intent(queries: list[dict]) -> str:
        """Infer intent label from query source_type(s)."""
        if not queries:
            return "education"

        source_type = queries[0].get("source_type", "")

        # Direct match in source type → intent mapping
        for intent, types in _INTENT_SOURCE_MAP.items():
            if source_type in types:
                return intent

        # Fuzzy: check if intent keyword appears in source_type
        for intent in _INTENT_SOURCE_MAP:
            if intent in source_type or source_type in intent:
                return intent

        return "education"
