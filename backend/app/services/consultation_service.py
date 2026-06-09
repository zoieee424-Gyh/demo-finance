from app.agents.compliance import ComplianceAgent
from app.agents.education import EducationAgent
from app.agents.financial_report import FinancialReportAgent
from app.agents.risk_control import RiskControlAgent
from app.rag.planner import RagPlanner
from app.rag.retriever import KnowledgeRetriever
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Intent
from app.services.advisory_agent_factory import get_advisory_agent
from app.services.compliance_guard import ComplianceGuard
from app.services.intent_router import IntentRouter


class ConsultationService:
    """Orchestrate the full consultation pipeline.

    Creation policy for the retriever:
      - If settings.rag_enabled=True AND a populated Chroma directory exists,
        KnowledgeRetriever is initialised with a ChromaStore for real semantic
        search.
      - In all other cases (rag disabled, Chroma dir missing, no data,
        no API key, dependency error) the retriever silently degrades to
        mock fallback.
      - This means the service NEVER crashes due to Chroma unavailability.

    Advisor mode (FIN_AGENT_ADVISOR_MODE):
      - "deepagent" (default): DeepAgentInvestmentAdvisor as primary.
          Falls back to InvestmentAdvisorAgent if DeepAgent unavailable.
      - "pipeline": Legacy InvestmentAdvisorAgent only.
      - "auto": DeepAgent-first, semantically same as "deepagent" for now.

    The advisory agent is obtained from the singleton factory — repeated
    ConsultationService instantiations share the same cached agent.
    """

    def __init__(self) -> None:
        from app.core.config import settings

        self.intent_router = IntentRouter()
        self.planner = RagPlanner()
        self.retriever = _create_retriever()
        self.guard = ComplianceGuard()

        self._advisor_mode = settings.advisor_mode
        self.advisory_agent = get_advisory_agent(self._advisor_mode)

        self.agents = {
            "advisory": self.advisory_agent,
            "financial_report": FinancialReportAgent(),
            "risk_control": RiskControlAgent(),
            "compliance": ComplianceAgent(),
            "education": EducationAgent(),
        }

    @property
    def advisor_mode(self) -> str:
        """The configured advisor mode (deepagent / pipeline / auto)."""
        return self._advisor_mode

    def handle(self, request: ConsultationRequest) -> ConsultationResponse:
        intent: Intent = self.intent_router.route(request.question)
        queries = self.planner.build_queries(request, intent)
        sources = self.retriever.retrieve(queries)
        response = self.agents[intent].answer(request, sources)
        return self.guard.review(response)


def _create_retriever() -> KnowledgeRetriever:
    """Build a retriever — Chroma-first when conditions are met, else mock-only.

    This function is intentionally conservative:
      1. Checks settings.rag_enabled (env: FIN_AGENT_RAG_ENABLED).
      2. Verifies the Chroma persist directory exists on disk (otherwise there
         is nothing to search).
      3. Creates an EmbeddingProvider and verifies DASHSCOPE_API_KEY is set
         (needed for query embedding).
      4. Creates a ChromaStore populated with at least one collection that
         has data.
      5. If ANY step fails or returns empty, returns a plain mock retriever.

    Returns:
        KnowledgeRetriever (with or without ChromaStore).
    """
    from app.core.config import settings

    if not settings.rag_enabled:
        return KnowledgeRetriever()

    try:
        return _try_create_chroma_retriever()
    except Exception:
        # Any failure → silent fallback to mock
        return KnowledgeRetriever()


def _try_create_chroma_retriever() -> KnowledgeRetriever:
    """Attempt to build a Chroma-backed retriever. May raise on any error."""
    import os

    from app.core.config import settings
    from app.rag.chroma_store import ChromaStore
    from app.rag.embedding_provider import EmbeddingProvider

    # ── 2. Chroma directory must exist ─────────────────────────────
    chroma_dir = settings.chroma_dir
    if not os.path.isdir(chroma_dir):
        return KnowledgeRetriever()

    # ── 3. Embedding provider must be configured ───────────────────
    provider = EmbeddingProvider(model=settings.embedding_model)
    if not provider.is_configured:
        return KnowledgeRetriever()

    # ── 4. Build store and verify at least one collection has data ─
    store = ChromaStore(persist_dir=chroma_dir, embedding_provider=provider)
    collections = store.list_collections()
    if not any(c.get("count", 0) > 0 for c in collections):
        return KnowledgeRetriever()

    return KnowledgeRetriever(chroma_store=store)
