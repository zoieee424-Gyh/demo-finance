"""Tests for /api/debug/risk-control endpoint."""

import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _block_deepagent(monkeypatch):
    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"
    monkeypatch.setattr("app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent", _noop_init)
    from app.services.advisory_agent_factory import reset_advisory_agent_cache
    reset_advisory_agent_cache()


def test_debug_returns_200():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/risk-control", json={
            "question": "评估组合风险",
            "user_profile": {
                "risk_preference": "conservative",
                "liquidity_need": "high",
                "holdings": [
                    {"asset_class": "权益类", "ratio": 70},
                    {"asset_class": "债券类", "ratio": 20},
                    {"asset_class": "现金及货币类", "ratio": 10},
                ],
            },
        })
    finally:
        settings.rag_enabled = old_rag

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "risk_control_direct"
    assert body["forced_intent"] == "risk_control"
    assert body["response"]["intent"] == "risk_control"


def test_debug_has_architecture():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/risk-control", json={"question": "评估风险"})
    finally:
        settings.rag_enabled = old_rag

    assert resp.status_code == 200
    arch = resp.json()["agent_architecture"]
    assert "configured_mode" in arch
    assert "fallback_used" in arch
    assert "actual_architecture" in arch
