import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import create_app


# ── Fixture: block real DeepAgent + reset cache ──────────────────

@pytest.fixture(autouse=True)
def _block_deepagent_and_reset_cache(monkeypatch):
    """Block real DeepAgent init and reset factory cache each test."""
    # Patch DeepAgentWrapper._init_deep_agent → no-op
    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"

    monkeypatch.setattr(
        "app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent",
        _noop_init,
    )
    # Clear the factory cache so each test gets a fresh agent
    from app.services.advisory_agent_factory import reset_advisory_agent_cache
    reset_advisory_agent_cache()


def test_debug_investment_advisor_endpoint_returns_advisory_response():
    old_rag_enabled = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/debug/investment-advisor",
            json={
                "question": "保守型投资者能买债券基金吗？风险大吗？",
                "user_profile": {"risk_preference": "low"},
            },
        )
    finally:
        settings.rag_enabled = old_rag_enabled

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "investment_advisor_direct"
    assert body["forced_intent"] == "advisory"
    assert body["rag_enabled"] is False
    assert body["retriever_has_chroma"] is False
    assert body["planned_queries"]
    assert body["evidence"]["item_count"] >= 1
    assert body["response"]["intent"] == "advisory"
    assert body["response"]["agent"] in ("investment_advisor", "investment_advisor_deepagent")
    assert "八、参考依据与适用边界" in body["response"]["answer"]
    assert body["response"]["risk_notice"]


def test_debug_investment_advisor_endpoint_keeps_no_stock_tip_boundary():
    old_rag_enabled = settings.rag_enabled
    settings.rag_enabled = False
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/debug/investment-advisor",
            json={"question": "能给我推荐几只股票吗？"},
        )
    finally:
        settings.rag_enabled = old_rag_enabled

    assert response.status_code == 200
    answer = response.json()["response"]["answer"]
    assert "推荐买入" not in answer
    assert "建议买入" not in answer
    assert "600" not in answer
    assert "000" not in answer


def test_debug_endpoint_returns_agent_architecture_info():
    """The debug endpoint must include agent_architecture metadata."""
    old_rag = settings.rag_enabled
    old_mode = settings.advisor_mode
    settings.rag_enabled = False
    settings.advisor_mode = "deepagent"
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/debug/investment-advisor",
            json={
                "question": "保守型投资者能买债券基金吗？",
                "user_profile": {"risk_preference": "low"},
            },
        )
    finally:
        settings.rag_enabled = old_rag
        settings.advisor_mode = old_mode

    assert response.status_code == 200
    body = response.json()
    assert "agent_architecture" in body, "Response missing agent_architecture field"
    arch = body["agent_architecture"]
    assert "configured_mode" in arch
    assert "agent_architecture" in arch
    assert "actual_architecture" in arch
    assert "deepagent_enabled" in arch
    assert "deepagent_available" in arch
    assert "fallback_used" in arch
    assert "tool_count" in arch
    # Default mode is "deepagent" — actual runtime depends on availability
    assert arch["configured_mode"] == "deepagent"
    # When DeepAgent is unavailable, actual_architecture falls back to "pipeline"
    assert arch["actual_architecture"] in ("deepagent", "pipeline")


def test_debug_endpoint_pipeline_mode_no_crash():
    """Under pipeline mode, debug endpoint must not crash."""
    old_rag = settings.rag_enabled
    old_mode = settings.advisor_mode
    settings.rag_enabled = False
    settings.advisor_mode = "pipeline"
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/debug/investment-advisor",
            json={
                "question": "帮我分析一下我的持仓是否合理？",
                "user_profile": {
                    "risk_preference": "conservative",
                    "holdings": [
                        {"asset_class": "宽基指数基金类", "ratio": 60},
                        {"asset_class": "债券类", "ratio": 40},
                    ],
                },
            },
        )
    finally:
        settings.rag_enabled = old_rag
        settings.advisor_mode = old_mode

    assert response.status_code == 200
    body = response.json()
    assert body["response"]["intent"] == "advisory"
    assert "六、持仓诊断" in body["response"]["answer"]


def test_cors_preflight_allows_local_frontend_origin():
    client = TestClient(create_app())
    response = client.options(
        "/api/debug/investment-advisor",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
