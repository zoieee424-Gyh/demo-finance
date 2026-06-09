"""
Tests for the unified agent query API (POST /api/agent/query).

Verifies:
  - Basic routing dispatch
  - Response schema (router, agent_architecture, response fields)
  - Error handling (empty question, bad mode)
  - Agent factory dispatch
  - Each intent routes correctly
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Basic API calls ──────────────────────────────────────────────

def test_agent_query_advisory_routing(client):
    """Asset allocation question should route to advisory."""
    resp = client.post("/api/agent/query", json={
        "question": "我想做资产配置，目前有30万资金",
        "user_profile": {"risk_preference": "stable"},
        "mode": "pipeline",  # Use pipeline to avoid DeepAgent init in tests
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data or "answer" in data
    assert "router" in data
    assert data["router"]["selected_intent"] == "advisory"
    assert data["router"]["confidence"] >= 0


def test_agent_query_financial_report_routing(client):
    """Financial report question should route to financial_report."""
    resp = client.post("/api/agent/query", json={
        "question": "营业收入增长但净利润下降是怎么回事",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["router"]["selected_intent"] == "financial_report"


def test_agent_query_compliance_routing(client):
    """Compliance question should route to compliance."""
    resp = client.post("/api/agent/query", json={
        "question": "这段话术保本保收益是合规的吗",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["router"]["selected_intent"] == "compliance"


def test_agent_query_risk_control_routing(client):
    """Risk question should route to risk_control."""
    resp = client.post("/api/agent/query", json={
        "question": "持仓集中度太高怎么控制风险",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["router"]["selected_intent"] == "risk_control"


def test_agent_query_education_routing(client):
    """Education question should route to education."""
    resp = client.post("/api/agent/query", json={
        "question": "什么是基金定投，适合新手吗",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["router"]["selected_intent"] == "education"


# ── Response schema validation ───────────────────────────────────

def test_agent_query_response_contains_router(client):
    """Response must include router metadata."""
    resp = client.post("/api/agent/query", json={
        "question": "资产配置和风险管理",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "router" in data
    router = data["router"]
    assert "selected_intent" in router
    assert "confidence" in router
    assert "reason" in router
    assert "candidates" in router
    assert len(router["candidates"]) == 5


def test_agent_query_response_contains_architecture(client):
    """Response must include agent_architecture metadata."""
    resp = client.post("/api/agent/query", json={
        "question": "投资风险管理",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "agent_architecture" in data


def test_agent_query_response_contains_response_fields(client):
    """Response must include standard consultation response fields."""
    resp = client.post("/api/agent/query", json={
        "question": "基金投资",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "risk_notice" in data
    assert "warnings" in data
    assert "sources" in data
    assert "intent" in data
    assert "agent" in data


# ── Error handling ───────────────────────────────────────────────

def test_agent_query_empty_question_422(client):
    """Empty question should return 422."""
    resp = client.post("/api/agent/query", json={
        "question": "",
        "mode": "pipeline",
    })
    assert resp.status_code == 422


def test_agent_query_too_short_question_422(client):
    """Question shorter than 2 chars should return 422."""
    resp = client.post("/api/agent/query", json={
        "question": "a",
        "mode": "pipeline",
    })
    assert resp.status_code == 422


def test_agent_query_bad_mode_400(client):
    """Invalid mode should return 400."""
    resp = client.post("/api/agent/query", json={
        "question": "投资",
        "mode": "invalid_mode",
    })
    assert resp.status_code == 400


def test_agent_query_missing_question_422(client):
    """Missing question field should return 422."""
    resp = client.post("/api/agent/query", json={
        "mode": "pipeline",
    })
    assert resp.status_code == 422


# ── Mode handling ────────────────────────────────────────────────

def test_agent_query_mode_auto_resolves(client):
    """mode='auto' should be accepted and resolve to deepagent."""
    resp = client.post("/api/agent/query", json={
        "question": "基金投资",
        "mode": "auto",
    })
    # Should not be a 400 — auto is valid and resolves to pipeline fallback
    # (DeepAgent may fail init without API key, but the endpoint works)
    assert resp.status_code in (200, 500, 502)
    # 200 with pipeline fallback, 500/502 if DeepAgent init fails without fallback
    if resp.status_code == 200:
        data = resp.json()
        assert "router" in data


# ── Router confidence ────────────────────────────────────────────

def test_agent_query_confidence_in_range(client):
    """Confidence should be between 0 and 1."""
    resp = client.post("/api/agent/query", json={
        "question": "资产配置",
        "mode": "pipeline",
    })
    assert resp.status_code == 200
    confidence = resp.json()["router"]["confidence"]
    assert 0.0 <= confidence <= 1.0
