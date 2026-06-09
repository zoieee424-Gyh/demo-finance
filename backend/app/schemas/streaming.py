"""
Streaming / SSE schemas for the debug stream API.

Provides:
  - PrepareRunRequest / PrepareRunResponse
  - StreamEventType literals
  - StreamEvent payload structure
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.consultation import ConsultationRequest

# ── Agent ID literals ────────────────────────────────────────────

AgentId = Literal[
    "advisory",
    "financial_report",
    "risk_control",
    "compliance",
    "education",
]

VALID_AGENT_IDS: set[str] = {
    "advisory", "financial_report", "risk_control", "compliance", "education",
}


# ── Prepare ──────────────────────────────────────────────────────

class PrepareRunRequest(BaseModel):
    """POST body for /api/debug/stream/{agent_id}/prepare."""
    question: str
    user_profile: dict[str, Any] | None = Field(default_factory=dict)


class PrepareRunResponse(BaseModel):
    """Returned after successful prepare."""
    run_id: str
    stream_url: str
    agent_id: str
    message: str = "Run prepared. Connect to stream_url with EventSource."


# ── Stream Event Types ───────────────────────────────────────────

StreamEventType = Literal[
    "run_started",
    "intent_selected",
    "retrieval_started",
    "retrieval_done",
    "agent_started",
    "heartbeat",
    "tool_started",
    "tool_done",
    "tool_failed",
    "tool_call",       # legacy — kept for compatibility
    "repair_applied",
    "fallback_used",
    "report_done",
    "run_done",
    "run_error",
]


# ── Stream Event ─────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """A single SSE event payload."""
    run_id: str = ""
    type: str  # StreamEventType
    stage: str = ""
    label: str = ""
    message: str = ""
    timestamp: str = ""
    # Optional enrichment
    tool_id: str | None = None
    tool_name: str | None = None
    tool_success: bool | None = None
    elapsed_ms: float | None = None
    # Final payload (run_done only)
    response: Any | None = None
    agent_architecture: Any | None = None
    evidence: Any | None = None
    planned_queries: list[Any] | None = None
