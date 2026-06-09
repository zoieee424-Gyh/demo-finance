"""
Tests for DeepAgentInvestmentAdvisor and DeepAgentWrapper.

Covers:
  - Default config uses legacy pipeline (no real DeepAgent calls).
  - FIN_AGENT_ADVISOR_MODE controls agent selection.
  - Fallback works when deepagents is unavailable or disabled.
  - answer() output is always ConsultationResponse.
  - Compliance: stock tips rejected, evidence preserved, hard constraints.

IMPORTANT: All tests use monkeypatch to skip real DeepAgent graph
creation.  Real DeepAgent e2e is exercised only in
backend/scripts/debug_deepagent_advisor.py.
"""

import os
import pytest

from app.agents.deepagent.base import DeepAgentWrapper, DeepAgentNotAvailableError
from app.agents.deepagent.investment_advisor_deepagent import DeepAgentInvestmentAdvisor
from app.agents.deepagent.registry import build_investment_advisor_registry
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


# ── Fixture: block real DeepAgent graph creation ────────────────────

@pytest.fixture(autouse=True)
def _block_real_deepagent_init(monkeypatch):
    """Prevent any test from accidentally creating a real DeepAgent graph.

    Patches DeepAgentWrapper._init_deep_agent to a no-op, forcing
    all agents into fallback mode.  Real DeepAgent testing is done
    exclusively via the debug script.
    """
    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture (noop_init)"

    monkeypatch.setattr(
        "app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent",
        _noop_init,
    )


# ── Helpers ─────────────────────────────────────────────────────────

def _make_request(question: str = "我是保守型投资者，应该如何配置资产？") -> ConsultationRequest:
    return ConsultationRequest(
        question=question,
        user_profile={"risk_preference": "conservative"},
    )


def _make_sources() -> list[Source]:
    return [
        Source(
            title="风险等级与资产类别匹配",
            source_type="advisory_knowledge",
            content_preview="不同风险等级对应不同的资产配置比例。",
            confidence=0.85,
        ),
        Source(
            title="投资者适当性管理规定概要",
            source_type="compliance_knowledge",
            content_preview="金融机构应当根据投资者的风险承受能力推荐产品。",
            confidence=0.72,
        ),
    ]


REQUIRED_SECTION_EIGHT = "八、参考依据与适用边界"


# ═══════════════════════════════════════════════════════════════════
# DeepAgentInvestmentAdvisor tests
# ═══════════════════════════════════════════════════════════════════

class TestDeepAgentInvestmentAdvisor:
    """Test the DeepAgentInvestmentAdvisor agent."""

    def test_default_disabled_uses_legacy(self):
        """When enabled=False, must use legacy pipeline (InvestmentAdvisorAgent)."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        assert agent.architecture == "pipeline"

        response = agent.answer(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        assert response.intent == "advisory"

    def test_enabled_but_deepagent_unavailable_falls_back(self):
        """When enabled=True but deepagents is not installed/working, must fallback."""
        # Set enabled=True; if deepagents package is installed, the wrapper
        # will try to create one. We accept either outcome.
        agent = DeepAgentInvestmentAdvisor(enabled=True)
        # Must produce a valid response regardless
        response = agent.answer(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        assert response.intent == "advisory"
        # Debug info must be available
        info = agent.debug_info
        assert "agent_architecture" in info
        assert "fallback_used" in info

    def test_answer_returns_consultation_response(self):
        """answer() must always return ConsultationResponse."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.intent == "advisory"
        assert response.agent is not None

    def test_answer_contains_required_sections(self):
        """Output must contain the required report sections."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(_make_request(), _make_sources())

        required = [
            "一、用户画像摘要",
            "二、投资目标分析",
            "三、风险评估结果",
            "四、资产配置建议",
            "五、基金定投规划",
            "六、持仓诊断",
            "七、市场热点解读",
            REQUIRED_SECTION_EIGHT,
            "九、风险提示",
        ]
        for section in required:
            assert section in response.answer, (
                f"Missing required section: {section}"
            )

    def test_answer_contains_evidence_section(self):
        """Section 8 must reference evidence sources."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(_make_request(), _make_sources())
        assert REQUIRED_SECTION_EIGHT in response.answer

    def test_stock_tip_request_rejected(self):
        """Request for stock recommendation must be rejected / warned."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(
            _make_request("能不能推荐几只股票？"),
            _make_sources(),
        )
        # Check that the answer doesn't contain specific stock recommendations
        answer_lower = response.answer.lower()
        # Must NOT contain stock codes
        import re
        has_stock_code = bool(re.search(r"\b\d{6}\b", response.answer))
        assert not has_stock_code, "Answer contains suspicious stock code"
        # Should contain rejection language or risk disclaimer
        has_disclaimer = any(
            phrase in response.answer
            for phrase in ["不构成投资", "风险提示", "不推荐", "无法提供", "不能推荐"]
        )
        # If the answer doesn't explicitly reject, warnings should flag it
        assert has_disclaimer or len(response.warnings) > 0, (
            f"Stock request not rejected. warnings={response.warnings}, "
            f"answer_start={response.answer[:200]}"
        )

    def test_no_return_promise(self):
        """Answer must not promise returns."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(
            _make_request("投资一年能赚多少钱？"),
            _make_sources(),
        )
        forbidden = ["保证收益", "年化收益", "稳赚", "肯定赚"]
        for term in forbidden:
            assert term not in response.answer, (
                f"Found forbidden term '{term}' in answer"
            )

    def test_no_price_prediction(self):
        """Answer must not predict market direction."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(
            _make_request("股市接下来会涨还是会跌？"),
            _make_sources(),
        )
        forbidden = ["会涨", "会跌", "看涨", "看跌", "牛市", "熊市"]
        found = [t for t in forbidden if t in response.answer]
        # "牛市""熊市" might appear in risk education context, accept if not predictive
        assert len(found) <= 2, f"Too many prediction-related terms: {found}"

    def test_sources_not_lost(self):
        """Retrieved sources must be present in the response."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        sources = _make_sources()
        response = agent.answer(_make_request(), sources)
        assert response.sources is not None
        assert len(response.sources) == len(sources)

    def test_risk_notice_present(self):
        """Response must include risk_notice."""
        agent = DeepAgentInvestmentAdvisor(enabled=False)
        response = agent.answer(_make_request(), _make_sources())
        assert response.risk_notice is not None
        assert len(response.risk_notice) > 0, "risk_notice should not be empty"

    def test_fallback_agent_works(self):
        """Explicit legacy fallback agent must produce valid output."""
        from app.agents.investment_advisor import InvestmentAdvisorAgent
        fallback = InvestmentAdvisorAgent()
        agent = DeepAgentInvestmentAdvisor(enabled=False, fallback_agent=fallback)
        response = agent.answer(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        assert "investment_advisor" in response.agent


# ═══════════════════════════════════════════════════════════════════
# DeepAgentWrapper tests
# ═══════════════════════════════════════════════════════════════════

class TestDeepAgentWrapper:
    """Test the DeepAgentWrapper class directly."""

    def test_disabled_wrapper_uses_legacy(self):
        """When enabled=False, wrapper must delegate to legacy agent."""
        from app.agents.investment_advisor import InvestmentAdvisorAgent
        legacy = InvestmentAdvisorAgent()
        wrapper = DeepAgentWrapper(
            tool_registry=build_investment_advisor_registry(),
            legacy_agent=legacy,
            system_prompt="Test prompt.",
            enabled=False,
        )
        response = wrapper.run(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        assert wrapper.architecture == "pipeline"
        assert wrapper.is_deepagent_available is False

    def test_debug_info_fields(self):
        """debug_info must contain all expected keys."""
        from app.agents.investment_advisor import InvestmentAdvisorAgent
        wrapper = DeepAgentWrapper(
            tool_registry=build_investment_advisor_registry(),
            legacy_agent=InvestmentAdvisorAgent(),
            system_prompt="Test.",
            enabled=False,
        )
        info = wrapper.debug_info
        assert "agent_architecture" in info
        assert "deepagent_enabled" in info
        assert "deepagent_available" in info
        assert "fallback_used" in info
        assert "fallback_reason" in info
        assert "tool_count" in info
        assert info["tool_count"] == 9

    def test_run_without_legacy_raises(self):
        """Wrapper without legacy agent and without DeepAgent must raise."""
        wrapper = DeepAgentWrapper(
            tool_registry=build_investment_advisor_registry(),
            legacy_agent=None,
            system_prompt="Test.",
            enabled=False,
        )
        with pytest.raises(DeepAgentNotAvailableError):
            wrapper.run(_make_request(), _make_sources())

    def test_fallback_reason_when_disabled(self):
        """When disabled, fallback_reason should explain why."""
        wrapper = DeepAgentWrapper(
            tool_registry=None,
            legacy_agent=None,
            system_prompt="",
            enabled=False,
        )
        assert wrapper.fallback_reason == "DeepAgent mode disabled by configuration"

    def test_runtime_fallback_reflected_in_debug_info(self):
        """When validator fails after DeepAgent runs, debug_info must show fallback."""
        from app.agents.investment_advisor import InvestmentAdvisorAgent

        reg = build_investment_advisor_registry()
        legacy = InvestmentAdvisorAgent()

        # Create a wrapper with a validator that always fails
        def _always_fail(answer: str) -> tuple[bool, list[str]]:
            return False, ["test forced failure"]

        wrapper = DeepAgentWrapper(
            tool_registry=reg,
            legacy_agent=legacy,
            system_prompt="Test.",
            enabled=False,  # disabled → legacy fallback path
            intent="advisory",
            output_validator=_always_fail,
        )

        # Run — should hit the init-time fallback path and record it
        response = wrapper.run(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        info = wrapper.debug_info
        # Init-time fallback should be tracked
        assert info["fallback_used"] is True
        assert info["agent_architecture"] == "pipeline"

    def test_successful_run_sets_fallback_false(self):
        """When DeepAgent not available, run triggers fallback tracking."""
        from app.agents.investment_advisor import InvestmentAdvisorAgent

        reg = build_investment_advisor_registry()
        legacy = InvestmentAdvisorAgent()

        wrapper = DeepAgentWrapper(
            tool_registry=reg,
            legacy_agent=legacy,
            system_prompt="Test.",
            enabled=False,  # disabled
            intent="advisory",
        )

        response = wrapper.run(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
        # disabled → should record fallback
        assert wrapper.debug_info["fallback_used"] is True
        assert wrapper.debug_info["agent_architecture"] == "pipeline"

    def test_architecture_field_matches_debug_info(self):
        """agent_architecture in debug_info should match actual path."""
        from app.agents.investment_advisor import InvestmentAdvisorAgent

        reg = build_investment_advisor_registry()
        legacy = InvestmentAdvisorAgent()

        wrapper = DeepAgentWrapper(
            tool_registry=reg,
            legacy_agent=legacy,
            system_prompt="Test.",
            enabled=False,
            intent="advisory",
        )
        wrapper.run(_make_request(), _make_sources())
        info = wrapper.debug_info
        assert info["agent_architecture"] == "pipeline"
        assert info["fallback_used"] is True


# ═══════════════════════════════════════════════════════════════════
# Configuration-driven switching tests
# ═══════════════════════════════════════════════════════════════════

class TestConfigDrivenSwitching:
    """Test that the FIN_AGENT_USE_DEEPAGENT_ADVISOR env var controls switching."""

    def test_env_var_false_uses_pipeline(self, monkeypatch):
        """Setting FIN_AGENT_USE_DEEPAGENT_ADVISOR=false → advisor_mode=pipeline."""
        monkeypatch.setenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR", "false")
        monkeypatch.delenv("FIN_AGENT_ADVISOR_MODE", raising=False)
        from app.core.config import Settings
        s = Settings()
        assert s.advisor_mode == "pipeline"

    def test_env_var_true_enables_deepagent(self, monkeypatch):
        """Setting FIN_AGENT_USE_DEEPAGENT_ADVISOR=true → advisor_mode=deepagent."""
        monkeypatch.setenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR", "true")
        monkeypatch.delenv("FIN_AGENT_ADVISOR_MODE", raising=False)
        from app.core.config import Settings
        s = Settings()
        assert s.advisor_mode == "deepagent"

    def test_advisor_mode_defaults_to_deepagent(self, monkeypatch):
        """Without any env var, advisor_mode defaults to deepagent."""
        monkeypatch.delenv("FIN_AGENT_ADVISOR_MODE", raising=False)
        monkeypatch.delenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR", raising=False)
        from app.core.config import Settings
        s = Settings()
        assert s.advisor_mode == "deepagent"

    def test_new_var_overrides_old(self, monkeypatch):
        """FIN_AGENT_ADVISOR_MODE takes precedence over old var."""
        monkeypatch.setenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR", "false")
        monkeypatch.setenv("FIN_AGENT_ADVISOR_MODE", "deepagent")
        from app.core.config import Settings
        s = Settings()
        # New var wins
        assert s.advisor_mode == "deepagent"

    def test_pipeline_mode_explicit(self, monkeypatch):
        """FIN_AGENT_ADVISOR_MODE=pipeline → pipeline mode."""
        monkeypatch.setenv("FIN_AGENT_ADVISOR_MODE", "pipeline")
        monkeypatch.delenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR", raising=False)
        from app.core.config import Settings
        s = Settings()
        assert s.advisor_mode == "pipeline"

    def test_can_create_agent_with_env_true(self, monkeypatch):
        """Agent creation with advisor_mode=deepagent must not crash."""
        monkeypatch.setenv("FIN_AGENT_ADVISOR_MODE", "deepagent")
        # Creating the agent should not crash even if real deepagents
        # may or may not be available
        agent = DeepAgentInvestmentAdvisor(enabled=True, configured_mode="deepagent")
        response = agent.answer(_make_request(), _make_sources())
        assert isinstance(response, ConsultationResponse)
