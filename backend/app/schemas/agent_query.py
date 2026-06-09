"""
Schemas for the unified agent query API.

POST /api/agent/query
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.consultation import ConsultationResponse, Intent


# ── Router candidate ──────────────────────────────────────────────

class RouterCandidate(BaseModel):
    """A single intent candidate with its score."""
    intent: str
    score: float


# ── Router decision ───────────────────────────────────────────────

class RouterDecision(BaseModel):
    """Routing result returned alongside the agent response."""
    selected_intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    candidates: list[RouterCandidate] = Field(default_factory=list)


# ── Agent architecture (reusable across debug + agent query) ─────

class ToolCallTraceItem(BaseModel):
    """Single tool invocation trace entry."""
    tool_id: str = ""
    name_cn: str = ""
    input_keys: list[str] = Field(default_factory=list)
    success: bool = True
    error_message: str | None = None
    output_preview: str | None = None
    elapsed_ms: float = 0.0


class AgentArchitectureInfo(BaseModel):
    """Agent architecture metadata for responses."""
    configured_mode: str = "deepagent"
    agent_architecture: str = "deepagent"
    actual_architecture: str = "deepagent"
    deepagent_enabled: bool = False
    deepagent_available: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    tool_count: int = 0
    tool_traces: list[ToolCallTraceItem] = Field(default_factory=list)


# ── Evidence debug (reusable across endpoints) ───────────────────

class EvidenceDebugItem(BaseModel):
    """Lightweight evidence item for debug display."""
    title: str = ""
    source_type: str = ""
    confidence: float = 0.0
    evidence_type: str = ""
    usage_hint: str = ""


class EvidenceDebugInfo(BaseModel):
    """Evidence pack summary for debug display."""
    item_count: int = 0
    has_advisory: bool = False
    has_risk: bool = False
    has_compliance: bool = False
    has_education: bool = False
    low_confidence: bool = True
    top_titles: list[str] = Field(default_factory=list)
    items: list[EvidenceDebugItem] = Field(default_factory=list)


# ── Request ──────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    """Unified agent query request — user inputs a question, system auto-routes."""
    question: str = Field(
        min_length=2, max_length=2000,
        description="User's natural-language financial question.",
    )
    user_profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional user profile with risk_preference, income_level, etc.",
    )
    mode: str = Field(
        default="deepagent",
        description="Agent execution mode: 'deepagent' | 'pipeline' | 'auto'. "
                    "'auto' is an alias for 'deepagent' in this build.",
    )


# ── Response ─────────────────────────────────────────────────────

class AgentQueryResponse(BaseModel):
    """Unified agent query response — includes routing metadata."""
    intent: str
    agent: str
    answer: str
    risk_notice: str
    sources: list[Any] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    agent_architecture: AgentArchitectureInfo | None = None
    router: RouterDecision | None = None
    rag_enabled: bool = False
    retriever_has_chroma: bool = False
    evidence: EvidenceDebugInfo | None = None
    planned_queries: list[dict] = Field(default_factory=list)
