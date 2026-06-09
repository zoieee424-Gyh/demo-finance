from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallTraceItem(BaseModel):
    """Single tool call trace entry."""
    tool_id: str = ""
    name_cn: str = ""
    input_keys: list[str] = Field(default_factory=list)
    success: bool = True
    error_message: str | None = None
    output_preview: str | None = None

from app.schemas.consultation import ConsultationResponse, Intent


class EvidenceDebugItem(BaseModel):
    title: str
    source_type: str
    confidence: float | None = None
    evidence_type: str
    usage_hint: str


class EvidenceDebugSummary(BaseModel):
    item_count: int
    has_advisory: bool
    has_risk: bool
    has_compliance: bool
    has_education: bool
    low_confidence: bool
    top_titles: list[str] = Field(default_factory=list)
    items: list[EvidenceDebugItem] = Field(default_factory=list)


class AgentArchitectureDebug(BaseModel):
    """DeepAgent architecture debug metadata."""
    # Value from FIN_AGENT_ADVISOR_MODE (deepagent / pipeline / auto)
    configured_mode: str = "deepagent"
    # Actual runtime architecture (deepagent / pipeline)
    agent_architecture: str = "pipeline"
    actual_architecture: str = "pipeline"
    deepagent_enabled: bool = False
    deepagent_available: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    tool_count: int = 0
    tool_traces: list[ToolCallTraceItem] = Field(default_factory=list)


class InvestmentAdvisorDebugResponse(BaseModel):
    mode: Literal["investment_advisor_direct"] = "investment_advisor_direct"
    forced_intent: Intent = "advisory"
    rag_enabled: bool
    retriever_has_chroma: bool
    agent_architecture: AgentArchitectureDebug = Field(default_factory=AgentArchitectureDebug)
    planned_queries: list[dict[str, Any]]
    evidence: EvidenceDebugSummary
    response: ConsultationResponse
