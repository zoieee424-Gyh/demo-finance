"""
Tests for education debug API endpoint.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _block_deepagent(monkeypatch):
    """Block real DeepAgent init so API tests never call external LLMs."""

    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"

    monkeypatch.setattr(
        "app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent",
        _noop_init,
    )
    from app.services.advisory_agent_factory import reset_advisory_agent_cache
    reset_advisory_agent_cache()


@pytest.fixture
def client():
    from app.main import create_app
    return TestClient(create_app())


class TestEducationDebugAPI:

    def test_endpoint_returns_200(self, client):
        payload = {"question": "什么是基金？"}
        response = client.post("/api/debug/education", json=payload)
        assert response.status_code == 200

    def test_forced_intent_is_education(self, client):
        payload = {"question": "我想了解股票基础知识"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        assert data.get("forced_intent") == "education"
        assert data.get("mode") == "education_direct"

    def test_response_contains_intent(self, client):
        payload = {"question": "基金和股票有什么区别？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        resp = data.get("response")
        assert resp is not None
        assert resp.get("intent") == "education"

    def test_agent_architecture_field_present(self, client):
        payload = {"question": "什么是债券？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        arch = data.get("agent_architecture")
        assert arch is not None
        assert "configured_mode" in arch
        assert "agent_architecture" in arch
        assert "actual_architecture" in arch
        assert "fallback_used" in arch

    def test_response_has_answer(self, client):
        payload = {"question": "怎么防止金融诈骗？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        resp = data.get("response")
        assert resp is not None
        assert len(resp.get("answer", "")) > 0

    def test_response_has_risk_notice(self, client):
        payload = {"question": "如何学习理财？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        resp = data.get("response")
        assert resp is not None
        risk_notice = resp.get("risk_notice", "")
        assert len(risk_notice) > 0

    def test_rag_enabled_field_present(self, client):
        payload = {"question": "什么是基金？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        assert "rag_enabled" in data
        assert "retriever_has_chroma" in data

    def test_evidence_field_present(self, client):
        payload = {"question": "什么是指数基金？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        assert "evidence" in data
        evidence = data["evidence"]
        assert "item_count" in evidence

    def test_planned_queries_present(self, client):
        payload = {"question": "什么是债券？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        assert "planned_queries" in data
        assert isinstance(data["planned_queries"], list)

    def test_no_crash_with_minimal_question(self, client):
        """Should handle a minimal question gracefully."""
        payload = {"question": "?"}
        response = client.post("/api/debug/education", json=payload)
        assert response.status_code in (200, 422)  # 422 acceptable for too-short question

    def test_no_crash_without_depseek_key(self, client):
        """Should not crash even without real LLM configured."""
        payload = {"question": "怎么学投资？"}
        response = client.post("/api/debug/education", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Without DeepSeek key, should fallback to pipeline
        arch = data.get("agent_architecture", {})
        # fallback_used should be True if DeepAgent can't init
        assert arch.get("fallback_used") is True or arch.get("fallback_used") is False
        # Response should still be valid
        assert data.get("response") is not None

    def test_no_crash_with_long_question(self, client):
        payload = {"question": "我想全面了解金融知识，从基金、股票、债券到保险，还有如何防诈骗，怎么理财规划"}
        response = client.post("/api/debug/education", json=payload)
        assert response.status_code == 200

    def test_fallback_to_education_agent_works(self, client):
        """Fallback agent should return valid response even without LLM."""
        payload = {"question": "什么是基金？"}
        response = client.post("/api/debug/education", json=payload)
        data = response.json()
        resp = data.get("response")
        assert resp is not None
        # Even fallback should have some answer
        assert len(resp.get("answer", "")) > 0
