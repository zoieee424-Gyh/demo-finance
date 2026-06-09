"""
DeepAgentInvestmentAdvisor — DeepAgent-powered investment advisory agent.

This agent replaces the hard-coded pipeline with a DeepAgent orchestrator
while keeping ALL deterministic financial tools intact.

Architecture:
  DeepAgent (LLM orchestrator)
    → profile_analyzer       (deterministic)
    → goal_planner           (deterministic)
    → risk_assessor          (deterministic)
    → allocation_engine      (deterministic)
    → fund_dca_planner       (deterministic)
    → holding_diagnostic     (deterministic, conditional)
    → market_hotspot_interpreter (deterministic, conditional)
    → evidence_builder       (deterministic)
    → advisory_compliance_policy (deterministic, post-hoc guaranteed)

Compliance:
  - All financial decisions come from deterministic tools, not the LLM.
  - Post-hoc compliance review is GUARANTEED regardless of DeepAgent behavior.
  - If the real DeepAgent is unavailable, the legacy pipeline is used instead.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import FinancialAgent
from app.agents.deepagent.base import (
    DeepAgentWrapper,
    INVESTMENT_ADVISOR_SYSTEM_PROMPT,
    validate_deepagent_output,
)
from app.agents.deepagent.registry import build_investment_advisor_registry
from app.agents.investment_advisor_tools.advisory_compliance_policy import (
    review as advisory_compliance_review,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


class DeepAgentInvestmentAdvisor(FinancialAgent):
    """DeepAgent-powered investment advisor.

    Follows the same FinancialAgent interface as the legacy InvestmentAdvisorAgent,
    so it can be dropped into ConsultationService without any other changes.

    Usage:
        agent = DeepAgentInvestmentAdvisor(
            enabled=True,
            fallback_agent=InvestmentAdvisorAgent(),
            configured_mode="deepagent",
        )
        response = agent.answer(request, sources)
    """

    name = "investment_advisor_deepagent"
    intent = "advisory"

    def __init__(
        self,
        *,
        enabled: bool = False,
        fallback_agent: FinancialAgent | None = None,
        configured_mode: str = "deepagent",
    ) -> None:
        """
        Args:
            enabled: Whether to attempt real DeepAgent creation.
            fallback_agent: Legacy InvestmentAdvisorAgent (or compatible).
                            Used when DeepAgent is unavailable or disabled.
            configured_mode: "deepagent" / "pipeline" / "auto" — the mode
                             requested via FIN_AGENT_ADVISOR_MODE.
        """
        self._enabled = enabled
        self._configured_mode = configured_mode

        # Lazy-import legacy agent to avoid circular imports
        if fallback_agent is None:
            from app.agents.investment_advisor import InvestmentAdvisorAgent
            fallback_agent = InvestmentAdvisorAgent()

        self._fallback_agent = fallback_agent
        self._registry = build_investment_advisor_registry()

        self._wrapper = DeepAgentWrapper(
            tool_registry=self._registry,
            legacy_agent=self._fallback_agent,
            system_prompt=INVESTMENT_ADVISOR_SYSTEM_PROMPT,
            agent_name="investment_advisor_deepagent",
            agent_name_cn="DeepAgent智能投顾编排器",
            enabled=self._enabled,
            intent="advisory",
            output_validator=validate_deepagent_output,
            compliance_review_func=advisory_compliance_review,
            risk_notice_default="投资有风险，入市需谨慎。本报告仅作信息参考，不构成投资决策依据。",
        )

    # ── Public API ──────────────────────────────────────────────────

    def answer(
        self,
        request: ConsultationRequest,
        sources: list[Source],
        event_callback: Any = None,
    ) -> ConsultationResponse:
        """Generate a compliance-checked advisory answer.

        Routes to the DeepAgent if available, otherwise to the legacy pipeline.
        """
        return self._wrapper.run(request, sources, event_callback=event_callback)

    # ── Debug / introspection ───────────────────────────────────────

    @property
    def architecture(self) -> str:
        """Current architecture: 'deepagent' or 'pipeline'."""
        return self._wrapper.architecture

    @property
    def configured_mode(self) -> str:
        """The mode requested via configuration (deepagent/pipeline/auto)."""
        return self._configured_mode

    @property
    def debug_info(self) -> dict:
        """Debug metadata for API responses."""
        info = self._wrapper.debug_info
        info["agent_name"] = self.name
        info["agent_name_cn"] = "DeepAgent智能投顾编排器"
        info["configured_mode"] = self._configured_mode
        return info
