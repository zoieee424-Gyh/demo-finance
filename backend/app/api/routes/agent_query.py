"""
Unified agent query endpoint — accepts any financial question,
auto-routes to the best agent, and returns a structured response.

POST /api/agent/query
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.rag.evidence import build_evidence_pack
from app.rag.planner import RagPlanner
from app.schemas.agent_query import (
    AgentArchitectureInfo,
    AgentQueryRequest,
    AgentQueryResponse,
    EvidenceDebugInfo,
    EvidenceDebugItem,
    RouterCandidate,
    RouterDecision,
    ToolCallTraceItem,
)
from app.schemas.consultation import (
    Source,
)
from app.services.agent_router import route_intent
from app.services.compliance_guard import ComplianceGuard
from app.services.consultation_service import _create_retriever

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent-query"])


# ── Agent dispatch ────────────────────────────────────────────────

def _get_agent_for_intent(intent: str, mode: str = "deepagent"):
    """Return a cached agent instance for the given intent.

    Uses the same factory pattern as debug endpoints — agents are
    cached at process level and reused across requests.
    """
    if intent == "advisory":
        from app.services.advisory_agent_factory import get_advisory_agent
        return get_advisory_agent(mode)
    elif intent == "financial_report":
        from app.services.financial_report_agent_factory import get_financial_report_agent
        return get_financial_report_agent(mode)
    elif intent == "risk_control":
        from app.services.risk_control_agent_factory import get_risk_control_agent
        return get_risk_control_agent(mode)
    elif intent == "compliance":
        from app.services.compliance_agent_factory import get_compliance_agent
        return get_compliance_agent(mode)
    elif intent == "education":
        from app.services.education_agent_factory import get_education_agent
        return get_education_agent(mode)
    else:
        raise ValueError(
            f"Unknown intent: {intent}. Must be one of: "
            "advisory, financial_report, risk_control, compliance, education."
        )


# ── Architecture info helpers ────────────────────────────────────

def _get_architecture_info(agent) -> AgentArchitectureInfo:
    """Collect agent architecture metadata for response."""
    info = agent.debug_info if hasattr(agent, "debug_info") else {}
    if info:
        raw_traces = info.get("tool_traces", [])
        tool_traces = [
            ToolCallTraceItem(
                tool_id=t.get("tool_id", ""),
                name_cn=t.get("name_cn", ""),
                input_keys=t.get("input_keys", []),
                success=t.get("success", True),
                error_message=t.get("error_message"),
                output_preview=t.get("output_preview"),
                elapsed_ms=t.get("elapsed_ms", 0.0),
            )
            for t in raw_traces
        ]
        arch = info.get("agent_architecture", "pipeline")
        return AgentArchitectureInfo(
            configured_mode=settings.advisor_mode,
            agent_architecture=arch,
            actual_architecture=arch,
            deepagent_enabled=info.get("deepagent_enabled", False),
            deepagent_available=info.get("deepagent_available", False),
            fallback_used=info.get("fallback_used", False),
            fallback_reason=info.get("fallback_reason"),
            tool_count=info.get("tool_count", 0),
            tool_traces=tool_traces,
        )
    return AgentArchitectureInfo(
        configured_mode=settings.advisor_mode,
        agent_architecture="pipeline",
        actual_architecture="pipeline",
        deepagent_enabled=False,
    )


def _build_evidence_debug(evidence_pack) -> EvidenceDebugInfo:
    """Build evidence debug summary from an EvidencePack."""
    return EvidenceDebugInfo(
        item_count=len(evidence_pack.items),
        has_advisory=evidence_pack.has_advisory,
        has_risk=evidence_pack.has_risk,
        has_compliance=evidence_pack.has_compliance,
        has_education=evidence_pack.has_education,
        low_confidence=evidence_pack.low_confidence,
        top_titles=evidence_pack.top_titles,
        items=[
            EvidenceDebugItem(
                title=item.title,
                source_type=item.source_type,
                confidence=item.confidence,
                evidence_type=item.evidence_type,
                usage_hint=item.usage_hint,
            )
            for item in evidence_pack.items
        ],
    )


# ── Endpoint ──────────────────────────────────────────────────────

@router.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(payload: AgentQueryRequest) -> AgentQueryResponse:
    """Unified agent query endpoint — auto-routes to the best agent.

    Accepts a free-form financial question and optional user profile.
    The system automatically determines whether to use the investment
    advisor, financial report analyzer, risk control reviewer,
    compliance reviewer, or education agent.

    Returns a structured consultation response with routing metadata.
    """
    question = payload.question.strip() if payload.question else ""
    if len(question) < 2:
        raise HTTPException(
            status_code=422,
            detail="question must have at least 2 characters",
        )

    if payload.mode not in ("deepagent", "pipeline", "auto"):
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: deepagent, pipeline, auto",
        )

    # Resolve mode: "auto" → "deepagent"
    mode = "deepagent" if payload.mode == "auto" else payload.mode

    # ── Step 1: Route intent ─────────────────────────────────────
    decision = route_intent(question, payload.user_profile)

    # ── Step 2: Get agent ────────────────────────────────────────
    try:
        agent = _get_agent_for_intent(decision.selected_intent, mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Step 3: Retrieval ────────────────────────────────────────
    planner = RagPlanner()
    retriever = _create_retriever()
    planned_queries = planner.build_queries(
        type(
            "FakeRequest", (),
            {"question": question, "user_profile": payload.user_profile},
        )(),
        decision.selected_intent,
    )
    sources: list[Source] = retriever.retrieve(planned_queries)

    # ── Step 4: Execute agent ────────────────────────────────────
    # Build a ConsultationRequest for the agent interface
    from app.schemas.consultation import ConsultationRequest
    req = ConsultationRequest(
        question=question,
        user_profile=payload.user_profile,
    )

    raw_response = agent.answer(req, sources)

    # ── Step 5: Compliance guard ─────────────────────────────────
    guard = ComplianceGuard()
    response = guard.review(raw_response)

    # ── Step 6: Evidence ─────────────────────────────────────────
    evidence_pack = build_evidence_pack(response.sources)

    # ── Step 7: Architecture metadata ────────────────────────────
    arch_info = _get_architecture_info(agent)

    return AgentQueryResponse(
        intent=response.intent,
        agent=response.agent,
        answer=response.answer,
        risk_notice=response.risk_notice,
        sources=[s.model_dump(mode="json") for s in response.sources],
        warnings=response.warnings,
        agent_architecture=arch_info,
        router=decision,
        rag_enabled=settings.rag_enabled,
        retriever_has_chroma=retriever.has_chroma,
        evidence=_build_evidence_debug(evidence_pack),
        planned_queries=planned_queries,
    )
