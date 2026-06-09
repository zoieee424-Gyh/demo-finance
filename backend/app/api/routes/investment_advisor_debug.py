from fastapi import APIRouter

from app.core.config import settings
from app.rag.evidence import build_evidence_pack
from app.rag.planner import RagPlanner
from app.schemas.advisory_debug import (
    AgentArchitectureDebug,
    EvidenceDebugItem,
    EvidenceDebugSummary,
    InvestmentAdvisorDebugResponse,
    ToolCallTraceItem,
)
from app.schemas.consultation import ConsultationRequest
from app.services.advisory_agent_factory import get_advisory_agent
from app.services.compliance_guard import ComplianceGuard
from app.services.consultation_service import _create_retriever


router = APIRouter(tags=["investment-advisor-debug"])


@router.post(
    "/debug/investment-advisor",
    response_model=InvestmentAdvisorDebugResponse,
)
def debug_investment_advisor(
    payload: ConsultationRequest,
) -> InvestmentAdvisorDebugResponse:
    """Run the investment advisor agent directly for API-level debugging.

    This endpoint bypasses intent routing on purpose: every request is handled
    as advisory so the first agent can be tested independently.
    """
    planner = RagPlanner()
    retriever = _create_retriever()

    # ── Use cached agent from factory — no per-request init ─────
    agent = get_advisory_agent(settings.advisor_mode)

    guard = ComplianceGuard()

    planned_queries = planner.build_queries(payload, "advisory")
    sources = retriever.retrieve(planned_queries)
    raw_response = agent.answer(payload, sources)
    response = guard.review(raw_response)
    evidence_pack = build_evidence_pack(response.sources)

    # ── Collect agent architecture info ─────────────────────────
    arch_info = _get_architecture_info(agent)

    return InvestmentAdvisorDebugResponse(
        rag_enabled=settings.rag_enabled,
        retriever_has_chroma=retriever.has_chroma,
        agent_architecture=arch_info,
        planned_queries=planned_queries,
        evidence=EvidenceDebugSummary(
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
        ),
        response=response,
    )


def _get_architecture_info(agent) -> AgentArchitectureDebug:
    """Collect agent architecture metadata for debug response."""
    if hasattr(agent, "debug_info"):
        info = agent.debug_info
        raw_traces = info.get("tool_traces", [])
        tool_traces = [
            ToolCallTraceItem(
                tool_id=t.get("tool_id", ""),
                name_cn=t.get("name_cn", ""),
                input_keys=t.get("input_keys", []),
                success=t.get("success", True),
                error_message=t.get("error_message"),
                output_preview=t.get("output_preview"),
            )
            for t in raw_traces
        ]
        arch = info.get("agent_architecture", "pipeline")
        return AgentArchitectureDebug(
            configured_mode=info.get("configured_mode", settings.advisor_mode),
            agent_architecture=arch,
            actual_architecture=arch,
            deepagent_enabled=info.get("deepagent_enabled", False),
            deepagent_available=info.get("deepagent_available", False),
            fallback_used=info.get("fallback_used", False),
            fallback_reason=info.get("fallback_reason"),
            tool_count=info.get("tool_count", 0),
            tool_traces=tool_traces,
        )
    # Legacy agent — no debug_info
    return AgentArchitectureDebug(
        configured_mode=settings.advisor_mode,
        agent_architecture="pipeline",
        actual_architecture="pipeline",
        deepagent_enabled=False,
    )
