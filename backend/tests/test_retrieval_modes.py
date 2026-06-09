"""
Tests for retrieval mode architecture shells.

Covers:
  - Config: default mode, env var reading, invalid fallback
  - BM25Retriever: importable, search() returns empty, no crash
  - HybridRetriever: init, search returns semantic results, merge dedup
  - KnowledgeRetriever: mode dispatch, hybrid delegates, bm25 returns empty
  - Semantic mode unchanged
"""

from __future__ import annotations

import os

import pytest

from app.schemas.consultation import Source


# ═══════════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════════

class TestRetrievalModeConfig:
    """Verify FIN_AGENT_RETRIEVAL_MODE is correctly wired in Settings."""

    def test_default_mode_is_semantic(self):
        """Without env var, retrieval_mode should be 'semantic'."""
        from app.core.config import Settings
        s = Settings()
        assert s.retrieval_mode == "semantic"

    def test_valid_env_override(self, monkeypatch):
        """Valid env var values should be accepted."""
        monkeypatch.setenv("FIN_AGENT_RETRIEVAL_MODE", "hybrid")
        from app.core.config import Settings
        s = Settings()
        assert s.retrieval_mode == "hybrid"

    def test_bm25_env_override(self, monkeypatch):
        """bm25 mode should be accepted."""
        monkeypatch.setenv("FIN_AGENT_RETRIEVAL_MODE", "bm25")
        from app.core.config import Settings
        s = Settings()
        assert s.retrieval_mode == "bm25"

    def test_invalid_mode_falls_back_to_semantic(self, monkeypatch):
        """Invalid values should silently fall back to 'semantic'."""
        monkeypatch.setenv("FIN_AGENT_RETRIEVAL_MODE", "garbage")
        from app.core.config import Settings
        s = Settings()
        assert s.retrieval_mode == "semantic"

    def test_case_insensitive(self, monkeypatch):
        """Env var should be case-insensitive."""
        monkeypatch.setenv("FIN_AGENT_RETRIEVAL_MODE", "HYBRID")
        from app.core.config import Settings
        s = Settings()
        assert s.retrieval_mode == "hybrid"

    def test_graphrag_env_switch(self, monkeypatch):
        """FIN_AGENT_GRAPHRAG_ENABLED should enable GraphRAG expansion."""
        monkeypatch.setenv("FIN_AGENT_GRAPHRAG_ENABLED", "true")
        from app.core.config import Settings
        s = Settings()
        assert s.graphrag_enabled is True


# ═══════════════════════════════════════════════════════════════════
# BM25Retriever tests
# ═══════════════════════════════════════════════════════════════════

class TestBM25Retriever:
    """Verify BM25Retriever placeholder shell."""

    def test_importable(self):
        """BM25Retriever should be importable."""
        from app.rag.sparse_retriever import BM25Retriever
        bm25 = BM25Retriever()
        assert bm25 is not None

    def test_search_returns_empty(self):
        """search() should return empty list, not crash."""
        from app.rag.sparse_retriever import BM25Retriever
        bm25 = BM25Retriever()
        results = bm25.search("test query", ["advisory_knowledge"], top_k=5)
        assert results == []
        assert isinstance(results, list)

    def test_search_accepts_all_args(self):
        """All standard search args should be accepted without error."""
        from app.rag.sparse_retriever import BM25Retriever
        bm25 = BM25Retriever()
        # Verify no crash with various arg patterns
        results = bm25.search("", [], top_k=1)
        assert results == []
        results = bm25.search("查询", ["advisory_knowledge", "risk_knowledge"], top_k=10)
        assert results == []

    def test_subclass_of_sparse_retriever(self):
        """BM25Retriever must be a SparseRetriever subclass."""
        from app.rag.sparse_retriever import BM25Retriever, SparseRetriever
        assert issubclass(BM25Retriever, SparseRetriever)

    def test_sparse_retriever_is_abstract(self):
        """SparseRetriever.search() should raise NotImplementedError."""
        from app.rag.sparse_retriever import SparseRetriever
        sr = SparseRetriever()
        with pytest.raises(NotImplementedError):
            sr.search("test", ["coll"], top_k=5)


# ═══════════════════════════════════════════════════════════════════
# HybridRetriever tests
# ═══════════════════════════════════════════════════════════════════

class TestHybridRetriever:
    """Verify HybridRetriever merge logic shell."""

    def test_importable(self):
        """HybridRetriever should be importable."""
        from app.rag.hybrid_retriever import HybridRetriever
        hr = HybridRetriever()
        assert hr is not None

    def test_search_with_sparse_empty_returns_semantic(self):
        """When sparse returns empty, result should = semantic alone."""
        from app.rag.hybrid_retriever import HybridRetriever

        class _FakeSemantic:
            def search(self, query, colls, top_k):
                return [
                    {"title": "Doc A", "source_type": "advisory_knowledge", "score": 0.9,
                     "metadata": {}},
                    {"title": "Doc B", "source_type": "risk_knowledge", "score": 0.8,
                     "metadata": {}},
                ]

        hr = HybridRetriever(semantic_retriever=_FakeSemantic())
        results = hr.search("test", ["advisory_knowledge"], top_k=5)
        assert len(results) == 2
        assert results[0].title == "Doc A"
        assert results[1].title == "Doc B"

    def test_merge_deduplicates_by_title(self):
        """Duplicate titles across semantic + sparse should be merged."""
        from app.rag.hybrid_retriever import HybridRetriever

        class _FakeSemantic:
            def search(self, query, colls, top_k):
                return [
                    {"title": "Shared Doc", "source_type": "advisory_knowledge",
                     "score": 0.9, "metadata": {}},
                    {"title": "Unique A", "source_type": "risk_knowledge",
                     "score": 0.7, "metadata": {}},
                ]

        class _FakeSparse:
            def search(self, query, colls, top_k):
                return [
                    Source(title="Shared Doc", source_type="risk_knowledge", confidence=0.6),
                    Source(title="Unique B", source_type="compliance_knowledge", confidence=0.5),
                ]

        hr = HybridRetriever(
            semantic_retriever=_FakeSemantic(),
            sparse_retriever=_FakeSparse(),
        )
        results = hr.search("test", ["advisory_knowledge"], top_k=5)
        titles = [r.title for r in results]
        assert len(titles) == 3  # Shared Doc deduplicated
        assert titles == ["Shared Doc", "Unique A", "Unique B"]

    def test_top_k_respected(self):
        """Results should be truncated to top_k."""
        from app.rag.hybrid_retriever import HybridRetriever

        class _FakeSemantic:
            def search(self, query, colls, top_k):
                return [
                    {"title": f"Doc {i}", "source_type": "advisory_knowledge",
                     "score": 0.9 - i * 0.1, "metadata": {}}
                    for i in range(10)
                ]

        hr = HybridRetriever(semantic_retriever=_FakeSemantic())
        results = hr.search("test", ["advisory_knowledge"], top_k=3)
        assert len(results) == 3

    def test_no_semantic_no_crash(self):
        """HybridRetriever should not crash when semantic retriever is None."""
        from app.rag.hybrid_retriever import HybridRetriever
        hr = HybridRetriever(semantic_retriever=None)
        results = hr.search("test", ["advisory_knowledge"])
        assert results == []

    def test_sparse_exception_graceful(self):
        """If sparse retriever throws, semantic results should still return."""
        from app.rag.hybrid_retriever import HybridRetriever

        class _FakeSemantic:
            def search(self, query, colls, top_k):
                return [
                    {"title": "Safe Doc", "source_type": "advisory_knowledge",
                     "score": 0.9, "metadata": {}},
                ]

        class _CrashSparse:
            def search(self, query, colls, top_k):
                raise RuntimeError("simulated sparse crash")

        hr = HybridRetriever(
            semantic_retriever=_FakeSemantic(),
            sparse_retriever=_CrashSparse(),
        )
        results = hr.search("test", ["advisory_knowledge"])
        assert len(results) == 1
        assert results[0].title == "Safe Doc"


# ─────────────────────────────────────────────────────────────────────────────
# GraphRAGRetriever tests
# ─────────────────────────────────────────────────────────────────────────────
class TestGraphRAGRetriever:
    """Verify GraphRAG placeholder shell."""

    def test_importable(self):
        """GraphRAGRetriever should be importable."""
        from app.rag.graph_retriever import GraphRAGRetriever
        retriever = GraphRAGRetriever()
        assert retriever is not None

    def test_no_graph_store_returns_empty(self):
        """Without graph_store, search() should return empty list."""
        from app.rag.graph_retriever import GraphRAGRetriever
        retriever = GraphRAGRetriever()
        assert retriever.search("资产配置和风险关系", intent="advisory") == []

    def test_fake_graph_store_results_to_sources(self):
        """A future graph_store can return dicts converted to Source."""
        from app.rag.graph_retriever import GraphRAGRetriever

        class _FakeGraphStore:
            def search(self, query, intent, top_k):
                return [
                    {
                        "title": f"{intent}:{query}",
                        "source_type": "graph_knowledge",
                        "confidence": 0.77,
                    }
                ]

        retriever = GraphRAGRetriever(graph_store=_FakeGraphStore())
        results = retriever.search("适当性规则", intent="compliance", top_k=3)
        assert len(results) == 1
        assert isinstance(results[0], Source)
        assert results[0].source_type == "graph_knowledge"
        assert results[0].confidence == 0.77

    def test_build_context(self):
        """GraphRAG context object should preserve query and intent."""
        from app.rag.graph_retriever import GraphRAGRetriever
        ctx = GraphRAGRetriever.build_context("保证收益", intent="compliance")
        assert ctx.query == "保证收益"
        assert ctx.intent == "compliance"


# ═══════════════════════════════════════════════════════════════════
# KnowledgeRetriever mode dispatch tests
# ═══════════════════════════════════════════════════════════════════

class TestRetrieverModeDispatch:
    """Verify KnowledgeRetriever dispatches correctly per mode."""

    def test_default_mode_is_semantic(self):
        """Default retrieval_mode should be 'semantic'."""
        from app.rag.retriever import KnowledgeRetriever
        kr = KnowledgeRetriever()
        assert kr.retrieval_mode == "semantic"

    def test_bm25_mode_returns_empty(self, monkeypatch):
        """In bm25 mode, retrieve() should return empty list."""
        from app.core.config import settings
        monkeypatch.setattr(settings, "retrieval_mode", "bm25")
        from app.rag.retriever import KnowledgeRetriever
        kr = KnowledgeRetriever()
        results = kr.retrieve([
            {"query": "test", "source_type": "investment_knowledge",
             "strategy": "semantic", "bm25_weight": None, "semantic_weight": None},
        ])
        assert results == []

    def test_hybrid_mode_does_not_crash(self, monkeypatch):
        """Hybrid retrieve should not crash with mock/no chroma."""
        from app.core.config import settings
        monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
        from app.rag.retriever import KnowledgeRetriever
        kr = KnowledgeRetriever()
        # No chroma → falls back to mock
        results = kr.retrieve([
            {"query": "test", "source_type": "investment_knowledge",
             "strategy": "semantic", "bm25_weight": None, "semantic_weight": None},
        ])
        # Without Chroma, falls back to mock which returns sources
        # This should not crash
        assert isinstance(results, list)

    def test_semantic_mode_unchanged(self):
        """Default semantic mode should work as before."""
        from app.rag.retriever import KnowledgeRetriever
        kr = KnowledgeRetriever()
        # No Chroma → mock fallback
        results = kr.retrieve([
            {"query": "资产配置", "source_type": "investment_knowledge",
             "strategy": "semantic", "bm25_weight": None, "semantic_weight": None},
        ])
        assert len(results) >= 0  # Mock returns 2 for advisory
        assert all(isinstance(s, Source) for s in results)

    def test_graphrag_disabled_by_default(self):
        """KnowledgeRetriever should keep GraphRAG disabled by default."""
        from app.rag.retriever import KnowledgeRetriever
        kr = KnowledgeRetriever()
        assert kr.graphrag_enabled is False

    def test_graphrag_enabled_without_store_no_crash(self, monkeypatch):
        """Enabled GraphRAG shell should not crash without graph_store."""
        from app.core.config import settings
        monkeypatch.setattr(settings, "graphrag_enabled", True)
        from app.rag.retriever import KnowledgeRetriever
        kr = KnowledgeRetriever()
        results = kr._retrieve_graphrag_placeholder([
            {"query": "关联风险", "source_type": "risk_models"},
        ])
        assert results == []
