"""Debug endpoint for ComplianceDeepAgent."""

from fastapi import APIRouter
from app.agents.deepagent.compliance_deepagent import ComplianceDeepAgent
from app.core.config import settings
from app.rag.evidence import build_evidence_pack
from app.rag.planner import RagPlanner
from app.schemas.advisory_debug import AgentArchitectureDebug, EvidenceDebugItem, EvidenceDebugSummary, ToolCallTraceItem
from app.schemas.compliance_debug import ComplianceDebugResponse
from app.schemas.consultation import ConsultationRequest
from app.services.compliance_guard import ComplianceGuard
from app.services.consultation_service import _create_retriever
from app.services.compliance_agent_factory import get_compliance_agent

router = APIRouter(tags=["compliance-debug"])


@router.post("/debug/compliance", response_model=ComplianceDebugResponse)
def debug_compliance(payload: ConsultationRequest) -> ComplianceDebugResponse:
    planner = RagPlanner()
    retriever = _create_retriever()
    # Use cached agent from factory — no per-request init
    agent = get_compliance_agent("deepagent")
    guard = ComplianceGuard()
    planned_queries = planner.build_queries(payload, "compliance")
    sources = retriever.retrieve(planned_queries)
    raw_response = agent.answer(payload, sources)
    response = guard.review(raw_response)
    evidence_pack = build_evidence_pack(response.sources)
    arch_info = _get_arch(agent)
    return ComplianceDebugResponse(
        rag_enabled=settings.rag_enabled, retriever_has_chroma=retriever.has_chroma,
        agent_architecture=arch_info, planned_queries=planned_queries,
        evidence=EvidenceDebugSummary(
            item_count=len(evidence_pack.items), has_advisory=evidence_pack.has_advisory,
            has_risk=evidence_pack.has_risk, has_compliance=evidence_pack.has_compliance,
            has_education=evidence_pack.has_education, low_confidence=evidence_pack.low_confidence,
            top_titles=evidence_pack.top_titles,
            items=[EvidenceDebugItem(title=item.title, source_type=item.source_type,
                confidence=item.confidence, evidence_type=item.evidence_type,
                usage_hint=item.usage_hint) for item in evidence_pack.items],
        ), response=response)


def _get_arch(agent) -> AgentArchitectureDebug:
    if hasattr(agent, "debug_info"):
        info = agent.debug_info
        traces = [ToolCallTraceItem(tool_id=t.get("tool_id",""), name_cn=t.get("name_cn",""),
            input_keys=t.get("input_keys",[]), success=t.get("success",True),
            error_message=t.get("error_message"), output_preview=t.get("output_preview"))
            for t in info.get("tool_traces", [])]
        arch = info.get("agent_architecture", "pipeline")
        return AgentArchitectureDebug(configured_mode=info.get("configured_mode","deepagent"),
            agent_architecture=arch, actual_architecture=arch,
            deepagent_enabled=info.get("deepagent_enabled",True),
            deepagent_available=info.get("deepagent_available",False),
            fallback_used=info.get("fallback_used",False),
            fallback_reason=info.get("fallback_reason"), tool_count=info.get("tool_count",0), tool_traces=traces)
    return AgentArchitectureDebug(configured_mode="deepagent", agent_architecture="pipeline", actual_architecture="pipeline")
