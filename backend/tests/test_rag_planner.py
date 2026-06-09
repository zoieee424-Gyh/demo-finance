"""Tests for RagPlanner."""
import pytest
from app.rag.planner import RagPlanner, _extract_key_phrases
from app.schemas.consultation import ConsultationRequest


class TestRagPlanner:

    @pytest.fixture
    def planner(self) -> RagPlanner:
        return RagPlanner()

    def test_build_queries_for_advisory(self, planner):
        """Advisory intent should produce queries with correct source types."""
        request = ConsultationRequest(question="我月薪1万，风险厌恶，3年后买房，该如何配置资产？")
        queries = planner.build_queries(request, "advisory")
        assert len(queries) >= 1, "Should have at least 1 query"
        assert queries[0]["query"] == request.question
        assert queries[0]["strategy"] == "semantic"
        assert "investment" in queries[0]["source_type"] or "knowledge" in queries[0]["source_type"]

    def test_build_queries_for_financial_report(self, planner):
        """Financial report intent should target report sources."""
        request = ConsultationRequest(question="分析贵州茅台2024年报的盈利能力")
        queries = planner.build_queries(request, "financial_report")
        assert len(queries) >= 1
        assert any("financial_reports" in q["source_type"] for q in queries)

    def test_all_queries_are_semantic_only(self, planner):
        """MVP: all queries should use semantic strategy, BM25/hybrid reserved."""
        request = ConsultationRequest(question="测试问题")
        for intent in ["advisory", "financial_report", "risk_control", "compliance", "education"]:
            queries = planner.build_queries(request, intent)
            for q in queries:
                assert q["strategy"] == "semantic"
                assert q["bm25_weight"] is None, "BM25 reserved, not implemented"
                assert q["semantic_weight"] is None, "semantic_weight reserved"

    def test_unknown_intent_falls_back(self, planner):
        """Unknown intent should still produce queries with general source."""
        request = ConsultationRequest(question="随便问个问题")
        queries = planner.build_queries(request, "unknown_intent")
        assert len(queries) >= 1
        assert queries[0]["source_type"] == "general_knowledge"


class TestPhraseExtraction:

    def test_extract_phrases(self):
        phrases = _extract_key_phrases("什么是基金 定投 如何操作")
        assert len(phrases) >= 1

    def test_empty_question(self):
        phrases = _extract_key_phrases("")
        assert phrases == []
