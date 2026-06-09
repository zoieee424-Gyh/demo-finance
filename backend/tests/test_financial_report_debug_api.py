"""Tests for /api/debug/financial-report endpoint."""

import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _block_deepagent(monkeypatch):
    """Block real DeepAgent init."""
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


def test_debug_endpoint_returns_200():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/financial-report", json={
            "question": "分析这份财报",
            "user_profile": {
                "financial_text": (
                    "某公司2024年实现营业收入120亿元，同比增长15%；"
                    "净利润18亿元，同比增长20%；资产负债率62%；"
                    "经营现金流为-3亿元。"
                ),
            },
        })
    finally:
        settings.rag_enabled = old_rag

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "financial_report_direct"
    assert body["forced_intent"] == "financial_report"
    assert body["response"]["intent"] == "financial_report"


def test_debug_response_has_architecture():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/financial-report", json={
            "question": "分析这份财报",
            "user_profile": {
                "financial_text": "某公司2024年营收100亿元，净利润15亿元。",
            },
        })
    finally:
        settings.rag_enabled = old_rag

    assert resp.status_code == 200
    arch = resp.json()["agent_architecture"]
    assert "configured_mode" in arch
    assert "agent_architecture" in arch
    assert "fallback_used" in arch


def test_debug_endpoint_rejects_stock_tip():
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/financial-report", json={
            "question": "推荐买入贵州茅台600519",
        })
    finally:
        settings.rag_enabled = old_rag

    assert resp.status_code == 200
    answer = resp.json()["response"]["answer"]
    assert "买入" not in answer or "不构成投资" in answer


def test_fallback_consistency_agent_matches_architecture():
    """If response.agent is legacy, fallback_used must be True."""
    old_rag = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        resp = client.post("/api/debug/financial-report", json={
            "question": "分析这份财报",
            "user_profile": {"financial_text": "某公司营收100亿元。"},
        })
    finally:
        settings.rag_enabled = old_rag

    assert resp.status_code == 200
    body = resp.json()
    agent = body["response"]["agent"]
    arch = body["agent_architecture"]

    # Consistency check: if legacy agent, fallback must be True
    if agent == "financial_report_analyst":
        assert arch["fallback_used"] is True, (
            f"Legacy agent '{agent}' but fallback_used=False. "
            f"fallback_reason={arch.get('fallback_reason')}"
        )
        assert arch["actual_architecture"] == "pipeline", (
            f"Legacy agent '{agent}' but actual_architecture="
            f"{arch.get('actual_architecture')}"
        )
    elif agent == "financial_report_deepagent":
        assert arch["fallback_used"] is False, (
            f"DeepAgent but fallback_used=True. "
            f"fallback_reason={arch.get('fallback_reason')}"
        )
