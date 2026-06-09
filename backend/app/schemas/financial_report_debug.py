"""Debug response schema for financial report analysis endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.advisory_debug import (
    AgentArchitectureDebug,
    EvidenceDebugItem,
    EvidenceDebugSummary,
)
from app.schemas.consultation import ConsultationResponse, Intent


class FinancialReportDebugResponse(BaseModel):
    mode: Literal["financial_report_direct"] = "financial_report_direct"
    forced_intent: Intent = "financial_report"
    rag_enabled: bool = False
    retriever_has_chroma: bool = False
    agent_architecture: AgentArchitectureDebug = Field(default_factory=AgentArchitectureDebug)
    planned_queries: list[dict[str, Any]] = Field(default_factory=list)
    evidence: EvidenceDebugSummary = Field(default_factory=lambda: EvidenceDebugSummary(
        item_count=0, has_advisory=False, has_risk=False,
        has_compliance=False, has_education=False, low_confidence=False,
    ))
    response: ConsultationResponse | None = None
