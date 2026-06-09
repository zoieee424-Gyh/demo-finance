"""Tests for /api/debug/compliance."""

import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _block(monkeypatch):
    def _noop(self):
        self._deep_agent_graph = None; self._deepagent_available = False; self._fallback_reason = "blocked"
    monkeypatch.setattr("app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent", _noop)
    from app.services.advisory_agent_factory import reset_advisory_agent_cache
    reset_advisory_agent_cache()


def test_returns_200():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/compliance", json={
            "question": "审查合规",
            "user_profile": {"content_to_review": "建议买入股票，目标价30元。", "scenario": "investment_advisory", "audience": "retail_investor"},
        })
    finally:
        settings.rag_enabled = old_rag
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "compliance_direct"
    assert body["response"]["intent"] == "compliance"


def test_has_architecture():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/compliance", json={"question": "审查合规"})
    finally:
        settings.rag_enabled = old_rag
    assert resp.status_code == 200
    arch = resp.json()["agent_architecture"]
    assert "fallback_used" in arch
    assert "actual_architecture" in arch
