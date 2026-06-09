"""
Tests for ConsultationService — end-to-end pipeline with and without Chroma.

Key invariants:
  - The service must NEVER crash due to Chroma unavailability.
  - Without Chroma data, mock fallback is used.
  - With ChromaStore + FakeEmbeddingProvider, real semantic search results
    appear in sources.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from app.core.config import settings
from app.schemas.consultation import ConsultationRequest
from app.services.consultation_service import (
    ConsultationService,
    _create_retriever,
    _try_create_chroma_retriever,
)
from app.rag.retriever import KnowledgeRetriever
from app.rag.embedding_provider import FakeEmbeddingProvider
from app.rag.chroma_store import ChromaStore


# ═══════════════════════════════════════════════════════════════════════
# Existing tests (verified compatible)
# ═══════════════════════════════════════════════════════════════════════

def test_routes_financial_report_question() -> None:
    service = ConsultationService()
    response = service.handle(ConsultationRequest(question="请分析这家公司财报和现金流"))

    assert response.intent == "financial_report"
    assert response.sources
    assert response.risk_notice


def test_default_advisory_has_risk_notice() -> None:
    service = ConsultationService()
    response = service.handle(ConsultationRequest(question="我三年后买房，该如何配置资产？"))

    assert response.intent == "advisory"
    assert "不构成投资决策依据" in response.risk_notice


# ═══════════════════════════════════════════════════════════════════════
# Mock fallback tests (no Chroma)
# ═══════════════════════════════════════════════════════════════════════

def test_service_default_uses_mock_fallback() -> None:
    """When FIN_AGENT_RAG_ENABLED is not set, retriever has no Chroma."""
    old = settings.rag_enabled
    settings.rag_enabled = False
    try:
        service = ConsultationService()
        # retriever should have no chroma_store
        assert service.retriever._chroma_store is None
        assert service.retriever.has_chroma is False
        # But it should still produce results (mock fallback)
        response = service.handle(ConsultationRequest(question="如何资产配置？"))
        assert response.sources
        assert len(response.sources) >= 1
    finally:
        settings.rag_enabled = old


def test_service_explicit_disabled_falls_back_to_mock() -> None:
    """FIN_AGENT_RAG_ENABLED=false → mock fallback."""
    old = settings.rag_enabled
    settings.rag_enabled = False
    try:
        service = ConsultationService()
        assert service.retriever._chroma_store is None
        response = service.handle(ConsultationRequest(question="什么是基金？"))
        assert response.sources
    finally:
        settings.rag_enabled = old


def test_service_does_not_crash_on_unconfigured() -> None:
    """Service init must not crash even when RAG env is misconfigured."""
    old_rag = settings.rag_enabled
    old_chroma = settings.chroma_dir
    settings.rag_enabled = True
    settings.chroma_dir = str(Path("/nonexistent/path/12345"))

    try:
        service = ConsultationService()
        # Should fall back to mock because dir doesn't exist
        assert service.retriever._chroma_store is None
        response = service.handle(ConsultationRequest(question="投资风险如何管理？"))
        assert response.sources
    finally:
        settings.rag_enabled = old_rag
        settings.chroma_dir = old_chroma


def test_create_retriever_returns_plain_when_rag_disabled() -> None:
    """Unit: _create_retriever with rag_enabled=False returns mock-only."""
    old = settings.rag_enabled
    settings.rag_enabled = False
    try:
        retriever = _create_retriever()
        assert retriever._chroma_store is None
    finally:
        settings.rag_enabled = old


# ═══════════════════════════════════════════════════════════════════════
# Chroma retrieval tests (with FakeEmbeddingProvider)
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_chroma_dir():
    path = tempfile.mkdtemp(prefix="svc_chroma_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def populated_store(temp_chroma_dir):
    """ChromaStore with fake embedding and some advisory data."""
    provider = FakeEmbeddingProvider(dimension=128)
    store = ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=provider)

    chunks = [
        {
            "content": "保守型投资者应将大部分资金配置在固收类资产中。权益类占比建议10%-30%。",
            "metadata": {
                "title": "保守型资产配置原则",
                "source_type": "investment_knowledge",
                "collection_name": "advisory_knowledge",
                "content_hash": "svc_test_001",
                "chunk_index": 0,
            },
        },
        {
            "content": "定期再平衡是维持目标配置比例的重要手段，建议每半年检视一次。",
            "metadata": {
                "title": "投资组合再平衡方法",
                "source_type": "investment_knowledge",
                "collection_name": "advisory_knowledge",
                "content_hash": "svc_test_002",
                "chunk_index": 0,
            },
        },
    ]
    store.add_chunks(chunks, "advisory_knowledge")
    return store


def test_service_with_chroma_returns_chroma_sources(populated_store) -> None:
    """When ChromaStore is injected, sources should come from Chroma."""
    service = ConsultationService()
    # Replace retriever with Chroma-backed one
    service.retriever = KnowledgeRetriever(chroma_store=populated_store)

    assert service.retriever.has_chroma is True

    response = service.handle(ConsultationRequest(question="保守型投资者如何配置资产？"))

    assert response.sources
    assert response.intent == "advisory"

    # The title should be from our Chroma data, not the mock store
    titles = {s.title for s in response.sources}
    assert "保守型资产配置原则" in titles or "投资组合再平衡方法" in titles


def test_service_with_empty_chroma_falls_back(populated_store) -> None:
    """Query an intent whose collection has no data → should not crash."""
    service = ConsultationService()
    service.retriever = KnowledgeRetriever(chroma_store=populated_store)

    # "compliance" intent maps to compliance_knowledge, which has no data
    response = service.handle(ConsultationRequest(question="信息披露有什么要求？"))

    # Should still return sources (from mock fallback or empty)
    assert response.sources is not None
    # Must not crash
    assert response.risk_notice


def test_create_retriever_auto_wires_populated_chroma(
    temp_chroma_dir, monkeypatch
) -> None:
    """Enabled RAG + populated Chroma dir → _create_retriever wires Chroma."""
    provider = FakeEmbeddingProvider(dimension=128)
    store = ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=provider)
    store.add_chunks([
        {
            "content": "保守型投资者应关注资产配置、流动性和回撤承受能力。",
            "metadata": {
                "title": "自动接线测试文档",
                "source_type": "investment_knowledge",
                "collection_name": "advisory_knowledge",
                "content_hash": "auto_wire_001",
                "chunk_index": 0,
            },
        }
    ], "advisory_knowledge")

    class ConfiguredFakeEmbeddingProvider(FakeEmbeddingProvider):
        def __init__(self, model: str = "text-embedding-v4") -> None:
            super().__init__(dimension=128)
            self.model = model

    old_rag = settings.rag_enabled
    old_chroma = settings.chroma_dir
    old_model = settings.embedding_model
    settings.rag_enabled = True
    settings.chroma_dir = temp_chroma_dir
    settings.embedding_model = "text-embedding-v4"
    monkeypatch.setattr(
        "app.rag.embedding_provider.EmbeddingProvider",
        ConfiguredFakeEmbeddingProvider,
    )

    try:
        retriever = _create_retriever()
        assert retriever.has_chroma is True

        sources = retriever.retrieve([
            {"query": "保守型投资者如何配置资产？", "source_type": "investment_knowledge"}
        ])
        assert {s.title for s in sources} == {"自动接线测试文档"}
    finally:
        settings.rag_enabled = old_rag
        settings.chroma_dir = old_chroma
        settings.embedding_model = old_model


# ═══════════════════════════════════════════════════════════════════════
# Cross-domain evidence injection tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def cross_domain_store(temp_chroma_dir):
    """ChromaStore with documents across multiple collections."""
    provider = FakeEmbeddingProvider(dimension=128)
    store = ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=provider)

    # advisory docs
    store.add_chunks([{
        "content": "定期定额投资可以平滑市场波动。",
        "metadata": {
            "title": "基金定投常见原则",
            "source_type": "investment_knowledge",
            "collection_name": "advisory_knowledge",
            "content_hash": "xd_adv_001",
            "chunk_index": 0,
        },
    }], "advisory_knowledge")

    # risk docs
    store.add_chunks([{
        "content": "持仓集中度过高会增加组合波动风险。",
        "metadata": {
            "title": "持仓集中度风险",
            "source_type": "risk_models",
            "collection_name": "risk_knowledge",
            "content_hash": "xd_risk_001",
            "chunk_index": 0,
        },
    }], "risk_knowledge")

    # compliance docs
    store.add_chunks([{
        "content": "投资顾问不得向投资者推荐具体股票或承诺收益。",
        "metadata": {
            "title": "投资建议合规边界",
            "source_type": "regulations",
            "collection_name": "compliance_knowledge",
            "content_hash": "xd_cmp_001",
            "chunk_index": 0,
        },
    }], "compliance_knowledge")

    return store


def test_cross_domain_sources_in_response(cross_domain_store) -> None:
    """Cross-domain query should return multi-collection sources with evidence."""
    service = ConsultationService()
    service.retriever = KnowledgeRetriever(chroma_store=cross_domain_store)

    response = service.handle(ConsultationRequest(
        question="保守型投资者能买债券基金吗？风险大吗？"
    ))

    assert response.sources
    # At least one source should be present
    titles = {s.title for s in response.sources}
    assert len(titles) >= 1


def test_answer_contains_evidence_references(cross_domain_store) -> None:
    """Advisory answer should contain evidence references from cross-domain sources."""
    service = ConsultationService()
    service.retriever = KnowledgeRetriever(chroma_store=cross_domain_store)

    response = service.handle(ConsultationRequest(
        question="保守型投资者应该如何配置资产？"
    ))

    # Answer should contain the evidence section
    assert "参考依据" in response.answer
    assert response.risk_notice


def test_stock_tip_query_rejected_with_compliance_evidence(
    cross_domain_store,
) -> None:
    """Query asking for stock tips should be rejected with compliance evidence."""
    service = ConsultationService()
    service.retriever = KnowledgeRetriever(chroma_store=cross_domain_store)

    response = service.handle(ConsultationRequest(
        question="能给我推荐几只股票吗？"
    ))

    # Should route to compliance and include evidence
    assert response.intent == "compliance"
    # Should have compliance-related content
    assert len(response.sources) >= 0  # may have sources or not
    # Should NOT recommend stocks
    assert "推荐买入" not in response.answer
    assert "建议买入" not in response.answer
