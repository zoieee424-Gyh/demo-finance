"""Debug response schema for risk control endpoint."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.advisory_debug import AgentArchitectureDebug, EvidenceDebugSummary
from app.schemas.consultation import ConsultationResponse, Intent


class RiskControlDebugResponse(BaseModel):
    mode: Literal["risk_control_direct"] = "risk_control_direct"
    forced_intent: Intent = "risk_control"
    rag_enabled: bool = False
    retriever_has_chroma: bool = False
    agent_architecture: AgentArchitectureDebug = Field(default_factory=AgentArchitectureDebug)
    planned_queries: list[dict[str, Any]] = Field(default_factory=list)
    evidence: EvidenceDebugSummary = Field(default_factory=lambda: EvidenceDebugSummary(
        item_count=0, has_advisory=False, has_risk=False,
        has_compliance=False, has_education=False, low_confidence=False,
    ))
    response: ConsultationResponse | None = None
