from typing import Any, Literal

from pydantic import BaseModel, Field


Intent = Literal["advisory", "financial_report", "risk_control", "compliance", "education"]


class Source(BaseModel):
    title: str
    source_type: str = "knowledge_base"
    url: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ConsultationRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    user_profile: dict[str, Any] = Field(default_factory=dict)


class ConsultationResponse(BaseModel):
    intent: Intent
    agent: str
    answer: str
    risk_notice: str
    sources: list[Source]
    warnings: list[str] = Field(default_factory=list)
