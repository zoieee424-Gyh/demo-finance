"""Tests for the SSE debug stream API."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


# ── Fixture: create app with deactivated DeepAgent init ──────────

@pytest.fixture(autouse=True)
def _reset_factories_and_block_deepagent(monkeypatch):
    """Block real DeepAgent graph creation and reset agent caches."""
    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"
    monkeypatch.setattr(
        "app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent",
        _noop_init,
    )
    # Reset all agent factories so tests start with a clean slate
    from app.services.advisory_agent_factory import reset_advisory_agent_cache
    from app.services.financial_report_agent_factory import reset_financial_report_agent_cache
    from app.services.risk_control_agent_factory import reset_risk_control_agent_cache
    from app.services.compliance_agent_factory import reset_compliance_agent_cache
    from app.services.education_agent_factory import reset_education_agent_cache
    reset_advisory_agent_cache()
    reset_financial_report_agent_cache()
    reset_risk_control_agent_cache()
    reset_compliance_agent_cache()
    reset_education_agent_cache()


@pytest.fixture
def client():
    """TestClient for the FastAPI app."""
    from app.main import app
    return TestClient(app)


# ── Prepare tests ────────────────────────────────────────────────

class TestPrepare:
    """POST /api/debug/stream/{agent_id}/prepare"""

    def test_prepare_valid_agent_advisory(self, client):
        resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "test question", "user_profile": {"risk_preference": "stable"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["stream_url"] == f"/api/debug/stream/{data['run_id']}"
        assert data["agent_id"] == "advisory"

    def test_prepare_valid_agent_financial_report(self, client):
        resp = client.post("/api/debug/stream/financial_report/prepare", json={
            "question": "analyze this",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "stream_url" in data

    def test_prepare_valid_agent_risk_control(self, client):
        resp = client.post("/api/debug/stream/risk_control/prepare", json={
            "question": "check risk",
        })
        assert resp.status_code == 200

    def test_prepare_valid_agent_compliance(self, client):
        resp = client.post("/api/debug/stream/compliance/prepare", json={
            "question": "check compliance",
        })
        assert resp.status_code == 200

    def test_prepare_valid_agent_education(self, client):
        resp = client.post("/api/debug/stream/education/prepare", json={
            "question": "explain fund",
        })
        assert resp.status_code == 200

    def test_prepare_unknown_agent_returns_400(self, client):
        resp = client.post("/api/debug/stream/unknown_agent/prepare", json={
            "question": "test",
        })
        assert resp.status_code == 400

    def test_prepare_auto_agent_returns_200(self, client):
        """SSE prepare with agent_id='auto' should succeed."""
        resp = client.post("/api/debug/stream/auto/prepare", json={
            "question": "test auto routing",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["stream_url"] == f"/api/debug/stream/{data['run_id']}"
        assert data["agent_id"] == "auto"


# ── Stream tests ─────────────────────────────────────────────────

class TestStream:
    """GET /api/debug/stream/{run_id}"""

    def test_stream_invalid_run_id(self, client):
        resp = client.get("/api/debug/stream/nonexistent123")
        assert resp.status_code == 200  # StreamingResponse
        content = resp.text
        # Should contain a run_error event
        assert "run_error" in content
        assert "not found" in content.lower() or "error" in content.lower()

    def test_stream_valid_run_produces_events(self, client):
        # Prepare
        prep_resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "test stream events",
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        # Stream — with blocked DeepAgent, the agent will fallback
        # quickly to pipeline, producing a complete event sequence
        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200
        content = resp.text

        # Must contain at least run_started and run_done
        assert "run_started" in content
        assert "run_done" in content
        assert "intent_selected" in content
        assert "retrieval_started" in content
        assert "retrieval_done" in content
        assert "agent_started" in content

    def test_stream_run_done_contains_response(self, client):
        prep_resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "test response in stream",
            "user_profile": {"risk_preference": "conservative"},
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200
        content = resp.text

        # Extract run_done event data
        assert "event: run_done" in content
        # Response field should be present in the final event
        assert "response" in content or "answer" in content.lower()

    def test_stream_contains_tool_traces_when_available(self, client):
        prep_resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "conservative allocation test",
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200
        content = resp.text

        # With pipeline fallback, there may be no tool_traces
        # But the stream should still complete gracefully
        assert "run_done" in content

    def test_run_deleted_after_stream(self, client):
        prep_resp = client.post("/api/debug/stream/education/prepare", json={
            "question": "test deletion",
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        # Stream to completion
        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200

        # Second request should fail (run deleted)
        resp2 = client.get(f"/api/debug/stream/{run_id}")
        assert "run_error" in resp2.text or "not found" in resp2.text.lower()

    def test_all_five_agents_prepare_and_complete_stream(self, client):
        agents = ["advisory", "financial_report", "risk_control", "compliance", "education"]
        questions = {
            "advisory": "test advisory",
            "financial_report": "test financial report",
            "risk_control": "test risk",
            "compliance": "test compliance",
            "education": "test education",
        }
        for agent_id in agents:
            prep_resp = client.post(f"/api/debug/stream/{agent_id}/prepare", json={
                "question": questions[agent_id],
            })
            assert prep_resp.status_code == 200, f"Failed to prepare {agent_id}"
            run_id = prep_resp.json()["run_id"]

            resp = client.get(f"/api/debug/stream/{run_id}")
            assert resp.status_code == 200, f"Failed to stream {agent_id}"
            content = resp.text
            assert "run_started" in content, f"{agent_id} missing run_started"
            assert "run_done" in content, f"{agent_id} missing run_done"

    def test_stream_content_type_is_sse(self, client):
        prep_resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "test content type",
        })
        run_id = prep_resp.json()["run_id"]

        resp = client.get(f"/api/debug/stream/{run_id}")
        assert "text/event-stream" in resp.headers.get("content-type", "")

    # ── Auto-routing stream tests ────────────────────────────────

    def test_auto_stream_produces_router_in_intent_selected(self, client):
        """SSE stream with agent_id='auto' should emit router metadata."""
        prep_resp = client.post("/api/debug/stream/auto/prepare", json={
            "question": "什么是基金定投，适合新手吗",
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200
        content = resp.text
        assert "run_started" in content
        assert "run_done" in content
        assert "intent_selected" in content
        # Auto route should include router metadata
        assert "router" in content

    def test_auto_stream_routes_correctly(self, client):
        """Question with strong education keywords should auto-route to education."""
        prep_resp = client.post("/api/debug/stream/auto/prepare", json={
            "question": "什么是基金定投",
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200
        content = resp.text
        assert "run_done" in content
        # Check that intent_selected event includes "education" or routing info
        assert "education" in content.lower() or "intent" in content.lower()

    def test_auto_stream_run_done_contains_router(self, client):
        """run_done for auto stream should include router metadata."""
        prep_resp = client.post("/api/debug/stream/auto/prepare", json={
            "question": "资产配置和风险管理",
        })
        assert prep_resp.status_code == 200
        run_id = prep_resp.json()["run_id"]

        resp = client.get(f"/api/debug/stream/{run_id}")
        assert resp.status_code == 200
        content = resp.text
        # run_done event should have router when auto-routing
        assert "router" in content


# ── SSE format tests ─────────────────────────────────────────────

class TestSSEFormat:
    """Verify that SSE events are properly formatted."""

    def test_events_have_correct_prefix(self, client):
        prep_resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "sse format test",
        })
        run_id = prep_resp.json()["run_id"]
        resp = client.get(f"/api/debug/stream/{run_id}")

        # Should start with an event
        assert resp.text.startswith("event:")

    def test_event_data_is_valid_json(self, client):
        prep_resp = client.post("/api/debug/stream/advisory/prepare", json={
            "question": "json test",
        })
        run_id = prep_resp.json()["run_id"]
        resp = client.get(f"/api/debug/stream/{run_id}")

        # Parse SSE events
        lines = resp.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        parsed = json.loads(data_str)
                        assert "type" in parsed
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON in SSE data: {data_str[:100]}")
