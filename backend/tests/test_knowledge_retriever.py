"""Tests for KnowledgeRetriever (mock + Chroma cross-domain)."""
import shutil
import tempfile

import pytest

from app.rag.retriever import KnowledgeRetriever
from app.rag.planner import RagPlanner
from app.rag.collection_selector import CollectionSelector
from app.rag.secondary_intent_detector import SecondaryIntentDetector
from app.schemas.consultation import ConsultationRequest, Source


class TestKnowledgeRetriever:

    @pytest.fixture
    def retriever(self) -> KnowledgeRetriever:
        return KnowledgeRetriever()

    @pytest.fixture
    def planner(self) -> RagPlanner:
        return RagPlanner()

    def test_retrieve_advisory_sources(self, retriever, planner):
        """Advisory queries should return investment knowledge sources."""
        request = ConsultationRequest(question="如何做资产配置？")
        queries = planner.build_queries(request, "advisory")
        sources = retriever.retrieve(queries)
        assert len(sources) >= 1
        for s in sources:
            assert isinstance(s, Source)
            assert s.title != ""
            assert s.source_type != ""
            assert 0.0 <= s.confidence <= 1.0

    def test_retrieve_education_sources(self, retriever, planner):
        """Education queries should return financial education sources."""
        request = ConsultationRequest(question="什么是市盈率？")
        queries = planner.build_queries(request, "education")
        sources = retriever.retrieve(queries)
        assert len(sources) >= 1
        # Education sources should exist
        titles = [s.title for s in sources]
        assert any("基金" in t or "PE" in t or "估值" in t for t in titles)

    def test_retrieve_deduplicates_by_title(self, retriever, planner):
        """Duplicate query source_types should not produce duplicate sources."""
        request = ConsultationRequest(question="资管新规和证券法")
        queries = planner.build_queries(request, "compliance")
        sources = retriever.retrieve(queries)
        titles = [s.title for s in sources]
        assert len(titles) == len(set(titles)), "Titles should be unique"

    def test_all_intents_return_sources(self, retriever, planner):
        """Every known intent should return at least 1 source."""
        for intent in ["advisory", "financial_report", "risk_control", "compliance", "education"]:
            request = ConsultationRequest(question="测试问题")
            queries = planner.build_queries(request, intent)
            sources = retriever.retrieve(queries)
            assert len(sources) >= 1, f"Intent '{intent}' should return sources"

    def test_empty_queries_returns_education_fallback(self, retriever):
        """Empty query list should still return something (education fallback)."""
        sources = retriever.retrieve([])
        assert len(sources) >= 1


class TestKnowledgeRetrieverCrossDomain:
    """Test cross-domain retrieval with Chroma + FakeEmbeddingProvider."""

    @pytest.fixture
    def temp_chroma_dir(self):
        path = tempfile.mkdtemp(prefix="chroma_xdomain_test_")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def fake_provider(self):
        from app.rag.embedding_provider import FakeEmbeddingProvider
        return FakeEmbeddingProvider(dimension=128)

    @pytest.fixture
    def chroma_store(self, temp_chroma_dir, fake_provider):
        from app.rag.chroma_store import ChromaStore
        return ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=fake_provider)

    def _add_doc(self, store, collection, title, content, source_type="knowledge_base"):
        """Helper to add one document chunk to a collection."""
        import hashlib
        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        chunks = [{
            "content": content,
            "metadata": {
                "title": title,
                "source_type": source_type,
                "chunk_index": 0,
                "content_hash": h,
                "token_count": len(content),
                "file_name": f"{title}.md",
                "format": "md",
            },
        }]
        store.add_chunks(chunks, collection)

    def test_cross_domain_retrieval_multi_collection(self, chroma_store):
        """Cross-domain query should search multiple collections."""
        # Populate two collections
        self._add_doc(chroma_store, "advisory_knowledge", "资产配置基础原则",
                      "资产配置是投资组合管理的核心环节。")
        self._add_doc(chroma_store, "risk_knowledge", "持仓集中度风险",
                      "持仓集中度过高会增加投资组合的波动风险。")

        selector = CollectionSelector()
        detector = SecondaryIntentDetector()
        retriever = KnowledgeRetriever(
            chroma_store=chroma_store,
            selector=selector,
            secondary_detector=detector,
        )

        # Query with both advisory and risk signals
        planner = RagPlanner()
        request = ConsultationRequest(question="如何配置资产同时控制风险？")
        queries = planner.build_queries(request, "advisory")
        sources = retriever.retrieve(queries)

        assert len(sources) >= 1
        titles = [s.title for s in sources]
        # May find either or both
        assert any("配置" in t or "集中度" in t or "风险" in t for t in titles)

    def test_secondary_detector_integrated(self, chroma_store):
        """KnowledgeRetriever uses SecondaryIntentDetector for collection expansion."""
        # Add docs to multiple collections
        self._add_doc(chroma_store, "advisory_knowledge", "基金定投常见原则",
                      "定期定额投资可以平滑市场波动带来的成本。")
        self._add_doc(chroma_store, "risk_knowledge", "权益类资产波动风险",
                      "权益类资产波动较大，投资者需要了解波动风险。")
        self._add_doc(chroma_store, "compliance_knowledge", "投资建议合规边界",
                      "投资顾问不得推荐具体股票或承诺收益。")

        retriever = KnowledgeRetriever(
            chroma_store=chroma_store,
            selector=CollectionSelector(),
            secondary_detector=SecondaryIntentDetector(),
        )

        planner = RagPlanner()
        # Query with risk words → should trigger secondary risk_control
        request = ConsultationRequest(question="定投有风险吗？合规吗？")
        queries = planner.build_queries(request, "advisory")
        sources = retriever.retrieve(queries)

        assert len(sources) >= 1

    def test_mock_fallback_unchanged(self):
        """Without Chroma, behaviour should be identical to before (mock-only)."""
        retriever = KnowledgeRetriever()
        sources = retriever.retrieve([{"query": "测试", "source_type": "investment_knowledge"}])
        assert len(sources) >= 1
        # Should return mock titles
        titles = [s.title for s in sources]
        assert any("资产配置" in t or "生命周期" in t for t in titles)
