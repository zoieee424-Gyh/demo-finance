"""
Tests for education agent factory caching.
"""
from __future__ import annotations

import pytest


# Block real DeepAgent init for all tests in this module
@pytest.fixture(autouse=True)
def _block_deepagent(monkeypatch):
    """Block real DeepAgent init so tests never call external LLMs."""

    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"

    monkeypatch.setattr(
        "app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent",
        _noop_init,
    )
    # Also block in education factory module
    from app.services.education_agent_factory import reset_education_agent_cache
    reset_education_agent_cache()
    from app.services.advisory_agent_factory import reset_advisory_agent_cache
    reset_advisory_agent_cache()


class TestEducationAgentFactory:

    def test_deepagent_mode_returns_cached_instance(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        a1 = get_education_agent("deepagent")
        a2 = get_education_agent("deepagent")
        assert a1 is a2

    def test_pipeline_mode_returns_education_agent(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        from app.agents.education import EducationAgent
        reset_education_agent_cache()
        agent = get_education_agent("pipeline")
        assert isinstance(agent, EducationAgent)

    def test_pipeline_mode_no_deepagent_init(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        agent = get_education_agent("pipeline")
        assert not hasattr(agent, "debug_info") or agent.debug_info.get(
            "deepagent_available", True
        ) is False

    def test_deepagent_mode_returns_financial_education_deep_agent(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        reset_education_agent_cache()
        agent = get_education_agent("deepagent")
        assert isinstance(agent, FinancialEducationDeepAgent)

    def test_auto_mode_equals_deepagent(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        reset_education_agent_cache()
        agent = get_education_agent("auto")
        assert isinstance(agent, FinancialEducationDeepAgent)
        assert agent.name == "financial_education_deepagent"

    def test_reset_clears_cache(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        a1 = get_education_agent("deepagent")
        reset_education_agent_cache()
        a2 = get_education_agent("deepagent")
        assert a1 is not a2

    def test_different_modes_different_instances(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        a_da = get_education_agent("deepagent")
        a_pl = get_education_agent("pipeline")
        assert a_da is not a_pl

    def test_agent_has_valid_intent(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        agent = get_education_agent("deepagent")
        assert agent.intent == "education"
        assert agent.name == "financial_education_deepagent"

    def test_warmup_returns_bool(self):
        from app.services.education_agent_factory import (
            warmup_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        result = warmup_education_agent("deepagent")
        assert isinstance(result, bool)

    def test_unknown_mode_falls_back(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        reset_education_agent_cache()
        agent = get_education_agent("invalid_mode")
        # Should fallback to deepagent
        assert agent.name == "financial_education_deepagent"

    def test_factory_independent_from_advisory(self):
        from app.services.education_agent_factory import (
            get_education_agent,
            reset_education_agent_cache,
        )
        from app.services.advisory_agent_factory import (
            get_advisory_agent,
            reset_advisory_agent_cache,
        )
        reset_education_agent_cache()
        reset_advisory_agent_cache()
        edu = get_education_agent("deepagent")
        adv = get_advisory_agent("deepagent")
        assert edu is not adv
        assert edu.intent == "education"
        assert adv.intent == "advisory"
